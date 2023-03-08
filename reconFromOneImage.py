import numpy as np
import twoViewTrian
import sqlite3
import test3DTo2D
import findPlane
import cv2

images = {}

def parseImageId(path):
    connection = sqlite3.connect(path+'database.db')
    cursor = connection.cursor()
    cursor.execute("SELECT image_id, camera_id, name FROM images;")
    for row in cursor:
        image_id = str(row[0])
        image_name = row[2]
        images[image_id] = image_name

def parseCamPara(cameras):
    camInPara = {}
    camOuPara = {}
    for image_id, image_name in images.items():
        # 内参
        K = np.zeros((3,3))
        K[0][0] = float(cameras[image_id][0])
        K[1][1] = float(cameras[image_id][0])
        K[2][2] = 1.0
        K[0][2] = float(cameras[image_id][1])
        K[1][2] = float(cameras[image_id][2])
        camInPara[image_id] = K

        # 外参
        MyRot = test3DTo2D.quat2Rot([float(cameras[image_id][3][0]),float(cameras[image_id][3][1]),float(cameras[image_id][3][2]),float(cameras[image_id][3][3])])
        P = np.zeros((3,4))
        P[0][0] = MyRot[0][0]
        P[0][1] = MyRot[0][1]
        P[0][2] = MyRot[0][2]
        P[1][0] = MyRot[1][0]
        P[1][1] = MyRot[1][1]
        P[1][2] = MyRot[1][2]
        P[2][0] = MyRot[2][0]
        P[2][1] = MyRot[2][1]
        P[2][2] = MyRot[2][2]

        P[0][3] = float(cameras[image_id][3][4])
        P[1][3] = float(cameras[image_id][3][5])
        P[2][3] = float(cameras[image_id][3][6])
        camOuPara[image_id] = P
    return camInPara,camOuPara



def readKeyPts(path):
    connection = sqlite3.connect(path+'database.db')
    cursor = connection.cursor()
    keypoints_all={}
    for image_id, image_name in images.items():
        cursor.execute("SELECT data FROM keypoints WHERE image_id=?;",
                       (image_id,))
        row = next(cursor)
        kp=[]
        if row[0] is None:
            keypoints = np.zeros((0, 6), dtype=np.float32)
        else:
            keypoints = np.frombuffer(row[0], dtype=np.float32).reshape(-1, 6)
            for r in range(keypoints.shape[0]):
                kp.append([keypoints[r,0],keypoints[r,1]])
        keypoints_all[image_id]=kp
    return keypoints_all

def reconFromOne(Kpts,camInPara,camOuPara,plane):
    with open("./out.txt","a+") as f:
        f.seek(0)
        f.truncate()
        idx = 0
        for image_id, image_name in images.items():
            print(f"{image_id}\n")
            hypo = 6.9035735
            # function 2: (a,b,c,d)*[R,T]^-1 *pt = 0
            planePara = np.zeros((1,4))
            planePara[0][0] = plane[0]
            planePara[0][1] = plane[1]
            planePara[0][2] = -1
            planePara[0][3] = plane[2]

            # T
            t = np.zeros((3,1))
            t[0][0] = camOuPara[image_id][0][3]
            t[1][0] = camOuPara[image_id][1][3]
            t[2][0] = camOuPara[image_id][2][3]

            # R
            r = np.zeros((3,3))
            r[0][0] = camOuPara[image_id][0][0]
            r[1][0] = camOuPara[image_id][1][0]
            r[2][0] = camOuPara[image_id][2][0]
            r[0][1] = camOuPara[image_id][0][1]
            r[1][1] = camOuPara[image_id][1][1]
            r[2][1] = camOuPara[image_id][2][1]
            r[0][2] = camOuPara[image_id][0][2]
            r[1][2] = camOuPara[image_id][1][2]
            r[2][2] = camOuPara[image_id][2][2]
            # # (A.T*A).I*A.T
            # func2Para = planePara.dot((np.linalg.pinv(camOuPara[image_id].T.dot(camOuPara[image_id]))).dot(camOuPara[image_id].T))
            # # RT-1
            # RT_ = (np.linalg.pinv(camOuPara[image_id].T.dot(camOuPara[image_id]))).dot(camOuPara[image_id].T)

            K = np.matrix(camInPara[image_id])
            K_ = K.I

            # function 1: pixPt = K*camPt / zc
            cnt = 0
            for u,v,_ in Kpts[image_id]:
                pixPt = np.zeros((3,1))
                pixPt[0][0] = u
                pixPt[1][0] = v
                pixPt[2][0] = 1.0
                camPt = K_.dot(pixPt) * hypo

                # wordPt3d
                wordPt3d = (r.T).dot(camPt - t)

                wordPt = np.zeros((4,1))
                wordPt[0][0] = wordPt3d[0][0]
                wordPt[1][0] = wordPt3d[1][0]
                wordPt[2][0] = wordPt3d[2][0]
                wordPt[3][0] = 1.0

                benchmark = planePara.dot(wordPt)

                # test satisfy
                if abs(benchmark) < 0.5:
                    # print(wordPt3d)
                    f.write(f"{idx} {wordPt3d[0][0]} {wordPt3d[1][0]} {wordPt3d[2][0]} \
                    {Kpts[image_id][cnt][2][2]} {Kpts[image_id][cnt][2][1]} {Kpts[image_id][cnt][2][0]} 0 1 0\n")
                cnt+=1

def parseCol(Kpts):
    for image_id, image_name in images.items():
        im = cv2.imread('./'+image_name)
        cnt = 0
        for u,v in Kpts[image_id]:
            Kpts[image_id][cnt].append(im[int(v)][int(u)])
            cnt+=1
    return Kpts


if __name__ == '__main__':
    parseImageId('./')
    # 解析相机
    cameras = twoViewTrian.readCamInfo('./')
    camInPara,camOuPara = parseCamPara(cameras)
    Kpts = readKeyPts('./')
    Kpts = parseCol(Kpts=Kpts)
    # print(Kpts)
    print("finish pt\n")
    x = findPlane.findPlane('./')
    reconFromOne(Kpts,camInPara,camOuPara,x)








