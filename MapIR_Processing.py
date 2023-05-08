# MapIR Band Correction

from MapIR import MapIRChromTest as mct
from MapIR import MapIRChromTestRAW as mctraw
from Hyperspectral_1_2 import Hyperspectral as hs
import Hyperspectral_1_2 as hype
from MapIR import MapIR_RAW as mraw
from MapIR import MapIR_jpg as mjpg
import MapIR as mir
import os



#------------------------------------------------
# Correction Process RAW:
#------------------------------------------------
file = 'RGN Files/AC 818/3.RAW'
image = mraw(file)
# image.display(hist=True)
image.NDVI(display=True, save=False)




#------------------------------------------------
# Correction Process JPG:
#------------------------------------------------
# file = 'RGN Files/818 Section.png'
# # file = 'RGN Files/818 Large Area.png'
# image = mjpg(file)
# image.correct()
# # image.display_image()
# image.NDVI(display=True, save=False)









