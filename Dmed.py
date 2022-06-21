from os import listdir
from os.path import isfile, join, exists

import gzip
import matplotlib.image as mpimg
import numpy as np

class Dmed():
    #Made by amr
    def __init__(self, dirIn):
        self.roiExt = '.jpg.ROI'
        self.imgExt = '.jpg'
        self.metaExt = '.meta'
        self.gndExt = '.GND'
        self.mapGzExt = '.map.gz'
        self.mapExt = '.map'
        self.baseDir = dirIn
        self.baseDir = dirIn
        self.data = []

        dirList = []

        for file in listdir(self.baseDir +'/'):
            if file.endswith(self.imgExt):
                self.data.append(file[:-4])

        self.origImgNum = len(self.data)
        self.imgNum = self.origImgNum - 1

    #Made by amr    
    def getNumOfImage(self):
        return self.imgNum + 1

    #Made by amr
    def getImg(self, id):
        if id < 0 or id > self.imgNum:
            img = []
            print('Index exceeds dataset size of ' + str(self.imgNum))
        else:
            imgAdress = self.baseDir + '/' + self.data[id] + self.imgExt
            img = mpimg.imread(imgAdress)
        return img

    #Made by abdulrehman
    def getONloc(self, id):
        onRow = []
        onCol = []
        if id < 0 or id > self.imgNum:
            print('Index exceeds dataset size of ' + str(self.imgNum))
        else:
            metaFile = self.baseDir + '/' + self.data[id] + self.metaExt
            with open(metaFile, 'r') as fMeta:
                if fMeta.fileno() > 0:
                    res = fMeta.read()
                    tokRow = res[res.find('~ONrow~')+7:]
                    onRow = int(tokRow[:tokRow.find('\n')])
                    tokCol = res[res.find('~ONcol~')+7:]
                    onCol = int(tokCol[:tokCol.find('\n')])
            return (onRow, onCol)

    # def getGT(self, id):
    #     if id < 0 or id > self.imgNum:
    #         img = []
    #         print('Index exceeds dataset size of ' + str(self.imgNum))
    #     else:
    #         mapGzFile = self.baseDir + '/' + self.data[id] + self.mapGzExt
    #         if exists(mapGzFile):
    #             gzip.decompress()