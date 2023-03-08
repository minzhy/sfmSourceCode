import sys
import numpy as np
import os
import sqlite3
import cv2
# 根据3D 查询2D 点
def test(point3Dtxt,image_path,database_path):
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()
    images = {}
    cursor.execute("SELECT image_id, camera_id, name FROM images;")
    for row in cursor:
        image_id = row[0]
        image_name = row[2]
        images[image_id] = image_name
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
        kp=np.array(kp)
        keypoints_all[image_name]=kp
    i=0
    with open(point3Dtxt,"r+") as f1:
        for line in f1.readlines():
            if i<3:
                i+=1
                continue
            value=line.strip().split()
            point3D_ID=int(value[0])
            track_imageAndkp_ID=value[8:]
            track_image_ID=[track_imageAndkp_ID[j] for j in range(len(track_imageAndkp_ID)) if j%2==0]
            track_kp_ID=[track_imageAndkp_ID[j] for j in range(len(track_imageAndkp_ID)) if j%2!=0]
            imgs=[]
            names=""
            for j in range(len(track_image_ID)):
                # print(images[int(track_image_ID[j])])
                # print()
                # print("\n")
                im=cv2.imread(os.path.join(image_path,images[int(track_image_ID[j])]))
                kp_all=keypoints_all[images[int(track_image_ID[j])]]
                kp=kp_all[int(track_kp_ID[j])]
                cv2.circle(im,(kp[0],kp[1]),50,(0,0,0),-1)
                print("keypoint location:{} {} point3D_ID:{} image_ID:{}".format(kp[0],kp[1],point3D_ID,track_image_ID[j]))
                imgs.append(im)
                im = cv2.resize(im,(400,400))
                cv2.imshow("1",im)
                cv2.waitKey(0)
                names+=images[int(track_image_ID[j])]
            temp=np.zeros(imgs[0].shape).astype("uint8")
            # for tt in imgs:
            #     temp=np.hstack((temp,tt))
            # cv2.namedWindow("{}".format(names),cv2.WINDOW_NORMAL)
            # cv2.imshow(names,temp)
            # cv2.waitKey(0)
            # cv2.imwrite(os.path.join(image_path,names)+".png",temp)
if __name__=="__main__":
    test(sys.argv[1],sys.argv[2],sys.argv[3])