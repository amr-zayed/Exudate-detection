from Dmed import Dmed
from exDetect import exDetect
import matplotlib.pyplot as plt


data = Dmed('./DMED')

for i in range(data.getNumOfImage()):
    
    rgbImg = data.getImg(i)
    
    onY, onX = data.getONloc(i) # get optic nerve location
    
    imgProb = exDetect(rgbImg, 1, onY, onX ); # segment exudates
    
    