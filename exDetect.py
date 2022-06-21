
from cv2 import imshow
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.color import rgb2hsv
from skimage.morphology import disk, erosion
from skimage import measure
from skimage.morphology import reconstruction
from scipy import signal

def exDetect( rgbImgOrig, removeON=1, onY=905, onX=290 ):
    
    #show lesions in image
    showRes = 1

    imgProb = getLesions( rgbImgOrig, showRes, removeON, onY, onX )

    return imgProb

#Mixed Contribution
def getLesions(rgbImgOrig, showRes, removeON, onY, onX):
    
    ##############################################################
    #                                                            #
    #                      Made By Amr                           #
    #                                                            #
    ##############################################################
    winOnRatio = np.array([1/8,1/8])

    origSize = np.shape(rgbImgOrig)
    newSize = (round(750*(origSize[1]/origSize[0])), 750)
    newSize = findGoodResolutionForWavelet(newSize)
    imgR = cv2.resize(rgbImgOrig[:,:,0], newSize)
    imgG = cv2.resize(rgbImgOrig[:,:,1], newSize)
    imgB = cv2.resize(rgbImgOrig[:,:,2], newSize)

    imgRGB = np.dstack((imgR,imgG, imgB))

    imgHSV = rgb2hsv(imgRGB)
    imgV = imgHSV[:,:,2]
    imgV8 = (imgV*255).astype(np.uint8)

    imgFovMask = getFovMask(imgV8, 1, 30)

    ##############################################################
    #                                                            #
    #                   Made By Abdulrehman                      #
    #                                                            #
    ##############################################################
    if removeON:
        onY = onY * newSize[0]/origSize[0]
        onX = onX * newSize[1]/origSize[1]
        onX = round(onX)
        onY = round(onY)
        winOnSize = np.round(np.multiply(winOnRatio, newSize))
        winOnCoordY = np.array([onY-winOnSize[0],onY+winOnSize[0]])
        winOnCoordX = np.array([onX-winOnSize[1],onX+winOnSize[1]])
        if winOnCoordY[0] < 0 : winOnCoordY[0]=0
        if winOnCoordX[0] < 0 : winOnCoordX[0]=0
        if winOnCoordY[1] > newSize[1] : winOnCoordY[1]=newSize[1]
        if winOnCoordX[1] > newSize[0] : winOnCoordY[1]=newSize[0]
        winOnCoordX = winOnCoordX.astype(int)
        winOnCoordY = winOnCoordY.astype(int)
    
    imgFovMask[winOnCoordY[0]:winOnCoordY[1], winOnCoordX[0]:winOnCoordX[1]] = 0

    ##############################################################
    #                                                            #
    #                      Made By Amr                           #
    #                                                            #
    ##############################################################

    medBG = np.double(signal.medfilt2d(imgV8, [round(newSize[1]/30), round(newSize[1]/30)]))

    maskImg = np.double(imgV8)
    pxLbl = maskImg < medBG
    maskImg[pxLbl] = medBG[pxLbl]

    medRestored = reconstruction(medBG, maskImg)

    subImg = np.double(imgV8) - np.double(medRestored)

    subImg = np.multiply(subImg, np.double(imgFovMask))
    subImg[subImg<0] = 0
    imgThNoOD = np.where(subImg==0, 0, 1)

    ##############################################################
    #                                                            #
    #                   Made By Abdulrehman                      #
    #                                                            #
    ##############################################################
    imgKirsch = kirschEdges(imgG)
    img0 = np.multiply(imgG, np.uint8(np.where(imgThNoOD == 0, 1,0)))
    
    img0recon = reconstruction(img0,imgG)
    img0kirsch = kirschEdges(img0recon)
    imgEdgeNoMask = imgKirsch - img0kirsch

    imgEdge = np.multiply (imgFovMask, imgEdgeNoMask)


    ##############################################################
    #                                                            #
    #                      Made By Amr                           #
    #                                                            #
    ##############################################################
    lbImgInfo = cv2.connectedComponentsWithStats((imgThNoOD).astype(np.uint8), 8, cv2.CV_32S)
    lbImg = lbImgInfo[1]
    r, c = newSize
    lesCandImg = np.zeros( (c,r) )
    lesCand = measure.regionprops(lbImg)
    
    for idxLes in range(len(lesCand)):
        pxIdxList = lesCand[idxLes].coords
        edgeSum = 0
        for coord in pxIdxList:
            edgeSum = edgeSum + imgEdge[int(coord[0]-1), int(coord[1]-1)]
        
        for coord in pxIdxList:
            lesCandImg[int(coord[0]-1), int(coord[1]-1)] = edgeSum / len(pxIdxList)


    ##############################################################
    #                                                            #
    #                   Made By Abdulrehman                      #
    #                                                            #
    ##############################################################
    lesCandImg = cv2.resize(lesCandImg,(origSize[1],origSize[0]),interpolation=cv2.INTER_NEAREST)
    if showRes :
        f = plt.figure()
        f.add_subplot(1,2, 1)
        plt.imshow(rgbImgOrig)
        f.add_subplot(1,2, 2)
        plt.imshow(lesCandImg)
        plt.show(block=True)
    return lesCandImg



##############################################################
#                                                            #
#                   Made By Abdulrehman                      #
#                                                            #
##############################################################
def findGoodResolutionForWavelet(sizeIn):
    maxWavDecom = 2
    pxToAddC = 2**maxWavDecom - sizeIn[1] % (2**maxWavDecom)
    pxToAddR = 2**maxWavDecom - sizeIn[0] % (2**maxWavDecom)
    sizeOut = (sizeIn[0] + pxToAddR , sizeIn[1] + pxToAddC)
    return sizeOut


##############################################################
#                                                            #
#                      Made By Amr                           #
#                                                            #
##############################################################
def getFovMask( gImg, erodeFlag=None, seSize=10 ):
    lowThresh = 0
    histRes = np.histogram(gImg, 256)[0]
    d = np.diff(histRes)
    d = np.where(d<lowThresh,0,1)
    
    fovMask = np.where(gImg>np.nonzero(d==1)[0][0],1,0)
    
    if erodeFlag and erodeFlag>0:
        fovMask = erosion(fovMask, disk(seSize))
        fovMask[0:seSize*2-1,:] = 0
        fovMask[:,0:seSize*2-1] = 0
        fovMask[-1-seSize*2:,:] = 0
        fovMask[:,-1-seSize*2:] = 0
    return fovMask


##############################################################
#                                                            #
#                   Made By Abdulrehman                      #
#                                                            #
##############################################################
def kirschEdges(imgIn):
    h1=np.array([[5, -3, -3],
        [5,  0, -3],
        [5, -3, -3]])/15
    h2=np.array([
        [-3, -3, 5],
        [-3,  0, 5],
        [-3, -3, 5]
    ])/15
    h3=np.array([
        [-3, -3,-3],
        [5,  0, -3],
        [5, 5, -3]
    ])/15
    h4=np.array([
        [-3, 5, 5],
        [-3,  0, 5],
        [-3, -3, -3]
    ])/15
    h5=np.array([
        [-3, -3, -3],
        [-3,  0, -3],
        [5, 5, 5]
    ])/15
    h6=np.array([
        [5, 5, 5],
        [-3,  0, -3],
        [-3, -3, -3],
    ])/15
    h7=np.array([
        [-3, -3, -3],
        [-3,  0, 5],
        [-3, 5, 5]
    ])/15
    h8=np.array([
        [5, 5, -3],
        [5,  0, -3],
        [-3, -3, -3]
    ])/15

    t1 = signal.convolve2d(imgIn,np.rot90(h1,2), mode='same')
    t2 = signal.convolve2d(imgIn,np.rot90(h2,2), mode='same')
    t3 = signal.convolve2d(imgIn,np.rot90(h3,2), mode='same')
    t4 = signal.convolve2d(imgIn,np.rot90(h4,2), mode='same')
    t5 = signal.convolve2d(imgIn,np.rot90(h5,2), mode='same')
    t6 = signal.convolve2d(imgIn,np.rot90(h6,2), mode='same')
    t7 = signal.convolve2d(imgIn,np.rot90(h7,2), mode='same')
    t8 = signal.convolve2d(imgIn,np.rot90(h8,2), mode='same')

    imgOut= np.maximum(t1,t2)
    imgOut = np.maximum(imgOut,t3)
    imgOut = np.maximum(imgOut,t4)
    imgOut = np.maximum(imgOut,t5)
    imgOut = np.maximum(imgOut,t6)
    imgOut = np.maximum(imgOut,t7)
    imgOut = np.maximum(imgOut,t8)

    return imgOut

