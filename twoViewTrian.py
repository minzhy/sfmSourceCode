import sys
import numpy as np
import os
import sqlite3
import cv2
import IDWM

def readCamInfo(path):
    cameras = {}
    with open(path+"cameras.txt") as f:
        for i in range(3):
            f.readline()
        camera = f.readline().split()
        while camera != []:
            # camera = camera.split()
            cameras[camera[0]] = [camera[4],camera[5],camera[6]]
            camera = f.readline().split()
    with open(path+'images.txt') as f:
        for i in range(4):
            f.readline()
        camera = f.readline().split()
        while camera != []:
            cameras[camera[0]].append([camera[1],camera[2],camera[3],camera[4],camera[5],camera[6],camera[7]])
            camera = f.readline()
            camera = f.readline().split()

    return cameras



if __name__=='__main__':
    readCamInfo('./')
    pass