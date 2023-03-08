##########IDWM triangulation############
#by licheng
import numpy as np
from numpy.linalg import norm
def AbsouteToRelative(R0,t0,R1,t1):
    '''
    R12 = R2 * R1.T
    '''
    R =  R1 * R0.T
    t = t1.reshape(3,-1) - R.dot( t0.reshape(3,-1))
    return R,t

def quat2Rot(q):
    MyRot = np.zeros((3,3))
    MyRot[0][0] = 1.0 - 2.0*q[2]*q[2] - 2.0*q[3]*q[3]
    MyRot[0][1] = 2.0*q[1]*q[2] - 2.0*q[0]*q[3]
    MyRot[0][2] = 2.0*q[1]*q[3] + 2.0*q[0]*q[2]
    MyRot[1][0] = 2.0*q[1]*q[2] + 2.0*q[0]*q[3]
    MyRot[1][1] = 1.0 - 2.0*q[1]*q[1] - 2.0*q[3]*q[3]
    MyRot[1][2] = 2.0*q[2]*q[3] - 2.0*q[0]*q[1]
    MyRot[2][0] = 2.0*q[1]*q[3] - 2.0*q[0]*q[2]
    MyRot[2][1] = 2.0*q[2]*q[3] + 2.0*q[0]*q[1]
    MyRot[2][2] = 1.0 - 2.0*q[1]*q[1] - 2.0*q[2]*q[2]
    print(f"myrot:{MyRot}")
    return MyRot

def TriangluateIDWMidPoint(P1,P2,point1,point2):
	# P1 : 3*4
	# P2 : 3*4
	# point1: 3*1
	# point2: 3*1
	# X: result
    R0 = P1[:,:3]
    t0 = P1[:,-1]
    R1 = P2[:,:3]
    t1 = P2[:,-1]
    R,t = AbsouteToRelative(R0,t0,R1,t1)
    Rx0 = np.dot(R ,point1)
    p_norm = norm(np.cross(Rx0.reshape(3) ,point2.reshape(3)))
    q_norm = norm(np.cross(Rx0.reshape(3),t.reshape(3)))
    r_norm = norm(np.cross(point2.reshape(3),t.reshape(3)))
    weight = q_norm /(q_norm + r_norm)
    X_ = weight * (t + (r_norm / p_norm)* (Rx0 + point2)) # equation 10
    X =R1.T.dot(X_ - t1.reshape(3,-1))
    lamda0_Rx0 = (r_norm / p_norm) * Rx0
    lamda1_x1 = (q_norm / p_norm) *point2
    # equation 9
    if (norm(t + lamda0_Rx0 - lamda1_x1))**2 < np.min([(norm(t + lamda0_Rx0 +lamda1_x1))**2,(norm(t - lamda0_Rx0 - lamda1_x1))**2,(norm(t-lamda0_Rx0+ lamda1_x1))**2]):
        return X ,True
    else:
        return [] ,False

if __name__=="__main__":
    # image ID:5 keypoint location:1647.774658203125 2743.65283203125
    pt1 = np.array([1647.774658203125,2743.65283203125,1]).reshape(3,-1)
    # 5672 3759 15341.000209830418 2836 1879.5
    pt1[0] = (pt1[0] - 2836) / 15341.000209830418
    pt1[1] = (pt1[1] - 1879.5) / 15341.000209830418

    # image Id:4 keypoint location:4481.7001953125 2713.098876953125
    pt2 = np.array([4481.7001953125,2713.098876953125,1]).reshape(3,-1)
    # 5672 3760 15326.993902170461 2836 1880
    pt2[0] = (pt2[0] - 2836) / 15326.993902170461
    pt2[1] = (pt2[1] - 1880) / 15326.993902170461

    # image Id:5 p1
    MyRot = quat2Rot([0.99995062496025888, 0.0098982426315984331, -0.00081221339850933274, -0.00033577341940723554])
    p1 = np.zeros((3,4))
    p1[0][0] = MyRot[0][0]
    p1[0][1] = MyRot[0][1]
    p1[0][2] = MyRot[0][2]
    p1[1][0] = MyRot[1][0]
    p1[1][1] = MyRot[1][1]
    p1[1][2] = MyRot[1][2]
    p1[2][0] = MyRot[2][0]
    p1[2][1] = MyRot[2][1]
    p1[2][2] = MyRot[2][2]
    # 2.1587826205368796 1.4140721884188354 0.17660303658800933
    p1[0][3] = 2.1587826205368796
    p1[1][3] = 1.4140721884188354
    p1[2][3] = 0.17660303658800933

    # image Id:4 p2
    MyRot = quat2Rot([0.99989746145777836 ,0.0092349803627336316 ,-0.010904887490571278 ,-0.000930127307721315])
    p2 = np.zeros((3,4))
    p2[0][0] = MyRot[0][0]
    p2[0][1] = MyRot[0][1]
    p2[0][2] = MyRot[0][2]
    p2[1][0] = MyRot[1][0]
    p2[1][1] = MyRot[1][1]
    p2[1][2] = MyRot[1][2]
    p2[2][0] = MyRot[2][0]
    p2[2][1] = MyRot[2][1]
    p2[2][2] = MyRot[2][2]
    # 3.5631452235567291 1.3884498367934142 0.24073758056883882
    p2[0][3] = 3.5631452235567291
    p2[1][3] = 1.3884498367934142
    p2[2][3] = 0.24073758056883882

    res = TriangluateIDWMidPoint(p1,p2,pt1,pt2)
    print(res)
    pass