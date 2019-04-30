"""
Created on Apr 21, 2019

Let's assume you want to rotate vectors around the origin of your
coordinate system. (If you want to rotate around some other point,
subtract its coordinates from the point you are rotating, do the
rotation, and then add back what you subtracted.) In 3D, you need
not only an angle, but also an axis. (In higher dimensions it gets
much worse, very quickly.)  Actually, you need 3 independent
numbers, and these come in a variety of flavors.  The flavor I
recommend is unit quaternions: 4 numbers that square and add up to
+1.  You can write these as [(x,y,z),w], with 4 real numbers, or
[v,w], with v, a 3D vector pointing along the axis. The concept
of an axis is unique to 3D. It is a line through the origin
containing all the points which do not move during the rotation.
So we know if we are turning forwards or back, we use a vector
pointing out along the line. Suppose you want to use unit vector u
as your axis, and rotate by 2t degrees.  (Yes, that's twice t.)
Make a quaternion [u sin t, cos t]. You can use the quaternion --
call it q -- directly on a vector v with quaternion
multiplication, q v q^-1, or just convert the quaternion to a 3x3
matrix M. If the components of q are {(x,y,z),w], then you want
the matrix

    M = {{1-2(yy+zz),  2(xy-wz),  2(xz+wy)},
         {  2(xy+wz),1-2(xx+zz),  2(yz-wx)},
         {  2(xz-wy),  2(yz+wx),1-2(xx+yy)}}.

Rotations, translations, and much more are explained in all basic
computer graphics texts.  Quaternions are covered briefly in
[Foley], and more extensively in several Graphics Gems, and the
SIGGRAPH 85 proceedings.

Refs:
[1] http://www.faqs.org/faqs/graphics/algorithms-faq/

@author: david
"""
import math

class Quaternion:
    def __init__(self):
        self.x, self.y, self.z = (0,0,0)
        self.w = 0
        
class _XYZTuple:
    def __init__(self, x=0, y=0, z=0):
        self.x, self.y, self.z = (x,y,z)
        
    @property
    def tuple(self):
        return tuple(self)
        
    def __iter__(self):
        return iter((self.x, self.y, self.z))    
        
    def __sub__(self, other):
        return Vector(*(x-y for x,y in zip(self, other)))
    
    def __str__(self):
        return "(%f,%f,%f)" % self.tuple
    
class Vector(_XYZTuple):
    pass

class Point(_XYZTuple):
    pass

class Matrix4x4:    
    W, X, Y, Z = (0, 1, 2, 3)
    def __init__(self, matrix=None):
        if matrix is not None:
            self.matrix = matrix
            return
        
        self.matrix = [[0]*4 for _ in range(4)]
        
    def __getitem__(self, key):
        return self.matrix[key]
    
    def rotation_only(self):
        import numpy as np
        
        return np.array([row[self.X:self.Z+1] 
                         for row in self.matrix[self.X:self.Z+1]]) 

    def from_axis_angle(self, axis, theta):
        """
        The following routine converts an angle and a unit axis vector
        to a matrix, returning the corresponding unit quaternion at no
        extra cost. It is written in such a way as to allow both fixed
        point and floating point versions to be created by appropriate
        definitions of FPOINT, ANGLE, VECTOR, QUAT, MATRIX, MUL, HALF,
        TWICE, COS, SIN, ONE, and ZERO.
        The following is an example of floating point definitions.
        
        #define FPOINT double
        #define ANGLE FPOINT
        #define VECTOR QUAT
        typedef struct {FPOINT x,y,z,w;} QUAT;
        enum Indices {X,Y,Z,W};
        typedef FPOINT MATRIX[4][4];
        
        Inputs
        ------
            axis : vector
            theta : angle
            m : 4x4 matrix
        Returns
        -------
            Quaternion
            m : 4x4 matrix
        """
        q = Quaternion()
        halfTheta = theta/2
        cosHalfTheta = math.cos(halfTheta)
        sinHalfTheta = math.sin(halfTheta)
    
        q.x = axis.x * sinHalfTheta
        q.y = axis.y * sinHalfTheta
        q.z = axis.z * sinHalfTheta
        q.w = cosHalfTheta;
        
        xs = 2*q.x  
        ys = 2*q.y  
        zs = 2*q.z
        wx = q.w * xs 
        wy = q.w * ys 
        wz = q.w * zs
        xx = q.x * xs 
        xy = q.x * ys 
        xz = q.x * zs
        yy = q.y * ys 
        yz = q.y * zs 
        zz = q.z * zs
        
        self.matrix[self.X][self.X] = 1 - (yy + zz)
        self.matrix[self.X][self.Y] = xy - wz
        self.matrix[self.X][self.Z] = xz + wy
        self.matrix[self.Y][self.X] = xy + wz
        self.matrix[self.Y][self.Y] = 1 - (xx + zz)
        self.matrix[self.Y][self.Z] = yz - wx
        self.matrix[self.Z][self.X] = xz - wy
        self.matrix[self.Z][self.Y] = yz + wx
        self.matrix[self.Z][self.Z] = 1 - (xx + yy)
        
        # Fill in remainder of 4x4 homogeneous transform matrix
        self.matrix[self.W][self.X] = 0
        self.matrix[self.W][self.Y] = 0
        self.matrix[self.W][self.Z] = 0
        self.matrix[self.X][self.W] = 0
        self.matrix[self.Y][self.W] = 0
        self.matrix[self.Z][self.W] = 0;
        self.matrix[self.W][self.W] = 1;
        return q
    
    @classmethod
    def from_any_axis_angle(cls, o, axis, theta, m):
        """
        The routine just given, MatrixFromAxisAngle, performs rotation about
        an axis passing through the origin, so only a unit vector was needed
        in addition to the angle. To rotate about an axis not containing the
        origin, a point on the axis is also needed, as in the following. For
        mathematical purity, the type POINT is used, but may be defined as:
        
        Input
        -----
            o : point
            axis : vector
            theta : angle
            m : 4x4 matrix
        Returns
        -------
            quaternion
            m : 4x4 matrix
        """
        q = m.from_axis_angle(axis, theta)
        m[cls.X][cls.W] = o.x-(m[cls.X][cls.X] 
                               * o.x+m[cls.X][cls.Y] 
                               * o.y+m[cls.X][cls.Z] 
                               * o.z)
        m[cls.Y][cls.W] = o.y-(m[cls.Y][cls.X] 
                               * o.x+m[cls.Y][cls.Y] 
                               * o.y+m[cls.Y][cls.Z] 
                               * o.z)
        m[cls.Z][cls.W] = o.x-(m[cls.Z][cls.X] 
                               * o.x+m[cls.Z][cls.Y] 
                               * o.y+m[cls.Z][cls.Z] 
                               * o.z)
        return q
    
    @staticmethod
    def from_points_angle(o, p, theta, m):
        """
        Inputs
        ------
            o : point
            p : point
            theta : angle
            m : 4x4 matrix
            
        Returns
        -------
            Quaternion
            m : 4x4 matrix
        """    
        axis = _normalize(p-o);
        return Matrix4x4.from_any_axis_angle(o, axis, theta, m);
    
def _normalize(v):
    """
    An axis can also be specified by a pair of points, with the direction
    along the line assumed from the ordering of the points. Although both
    the previous routines assumed the axis vector was unit length without
    checking, this routine must cope with a more delicate possibility. If
    the two points are identical, or even nearly so, the axis is unknown.
    For now the auxiliary routine which makes a unit vector ignores that.
    
    Inputs
    ------
        v : vector
        
    Returns
    -------
        vector
    """
    
    norm = v.x**2 + v.y**2 + v.z**2
    
    # Better to test for (near-)zero norm before taking reciprocal.
    scl = 1/math.sqrt(norm)
    u = Vector(*(x * scl for x in v))
    return u
    
