import numpy as np

# 0.99989746145777836 0.0092349803627336316 -0.010904887490571278 -0.000930127307721315 3.5631452235567291 1.3884498367934142 0.24073758056883882
# 5 0.99995062496025888 0.0098982426315984331 -0.00081221339850933274 -0.00033577341940723554 2.1587826205368796 1.4140721884188354 0.17660303658800933 5 00004.jpg

def quat2Rot(q):
    # MyEuler = R.from_quat([q]).as_euler('zyx')
    # # print(MyEuler)
    # MyRot = R.from_euler('zyx',MyEuler)
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

if __name__=='__main__':
    # parameter input
    MyRot = quat2Rot([0.99989746145777836, 0.0092349803627336316, -0.010904887490571278, -0.000930127307721315])
    # 4 image
#     [  0.9997604,  0.0016587, -0.0218247;
#   -0.0020615,  0.9998277, -0.0184478;
#    0.0217904,  0.0184884,  0.9995916 ]

    # 5 image
# [  0.9999985,  0.0006554, -0.0016310;
#   -0.0006876,  0.9998038, -0.0197950;
#    0.0016177,  0.0197961,  0.0197961 ]
    # 仅仅和相机的内参的四元数有关系
    MyRot = np.zeros((1,3,3))
    MyRot[0][0][0] = 0.9999985
    MyRot[0][0][1] = 0.0006554
    MyRot[0][0][2] = -0.0016310
    MyRot[0][1][0] = -0.0006876
    MyRot[0][1][1] = 0.9998038
    MyRot[0][1][2] = -0.0197950
    MyRot[0][2][0] = 0.0016177
    MyRot[0][2][1] = 0.0197961
    MyRot[0][2][2] = 0.9998027

    # 相机外参
    outParametersOfCam = np.zeros((4,4))
    outParametersOfCam[0][0] = MyRot[0][0][0]
    outParametersOfCam[0][1] = MyRot[0][0][1]
    outParametersOfCam[0][2] = MyRot[0][0][2]
    outParametersOfCam[1][0] = MyRot[0][1][0]
    outParametersOfCam[1][1] = MyRot[0][1][1]
    outParametersOfCam[1][2] = MyRot[0][1][2]
    outParametersOfCam[2][0] = MyRot[0][2][0]
    outParametersOfCam[2][1] = MyRot[0][2][1]
    outParametersOfCam[2][2] = MyRot[0][2][2]

    outParametersOfCam[0][3] = 2.1587826205368796
    outParametersOfCam[1][3] = 1.4140721884188354
    outParametersOfCam[2][3] = 0.17660303658800933

    outParametersOfCam[3][3] = 1.0

    # 相机内参
    inParametersOfCam = np.zeros((3,3))
    inParametersOfCam[0][0] = 15341.000209830418
    inParametersOfCam[1][1] = 15341.000209830418
    inParametersOfCam[2][2] = 1.0
    inParametersOfCam[0][2] = 2836.0
    inParametersOfCam[1][2] = 1879.5

    ptInC = np.zeros((4,1))
    ptInC = outParametersOfCam.dot(np.array([[-3.4076628302337939],[-0.73467474040161374],[6.7246727017340842],[1]])) # 世界坐标系下的点
    print(ptInC)
    z = ptInC[2]
    ptInC3d = np.zeros((3,1))
    ptInC3d[0] = ptInC[0]
    ptInC3d[1] = ptInC[1]
    ptInC3d[2] = ptInC[2]
    ptInC3d.reshape(1,-1)

    pixPt = np.zeros((3,1))
    pixPt = inParametersOfCam.dot(ptInC3d) / z
    print(pixPt)
    pass
