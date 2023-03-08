import numpy as np

pts = []

def findPlane(path):
    Adata = []
    Bdata = []
    with open(path+'points3D.txt') as f:
        for i in range(3):
            line = f.readline()
        line = f.readline().split()
        while line != []:
            pts.append([line[1],line[2],line[3]])
            Adata.extend([float(line[1]),float(line[2]),1.0])
            Bdata.extend([float(line[3])])
            line = f.readline().split()
    A = np.matrix(Adata).reshape(len(pts),3)
    B = np.matrix(Bdata).reshape(len(pts),1)
    x = (A.T*A).I*A.T*B
    return x

def testMaxAndMinDim(path):
    minZ = float('inf')
    maxZ = float('-inf')
    minX = float('inf')
    maxX = float('-inf')
    minY = float('inf')
    maxY = float('-inf')
    with open(path+'points3D.txt') as f:
        for i in range(3):
            line = f.readline()
        line = f.readline().split()
        while line != []:
            minZ = min(minZ,float(line[3]))
            maxZ = max(maxZ,float(line[3]))
            minX = min(minX,float(line[1]))
            maxX = max(maxX,float(line[1]))
            minY = min(minY,float(line[2]))
            maxY = max(maxY,float(line[2]))
            line = f.readline().split()
    return (minX,maxX,minY,maxY,minZ,maxZ)


if __name__ == '__main__':
    x = findPlane('./')
    print(x)
    k = testMaxAndMinDim('./')
    print(k)