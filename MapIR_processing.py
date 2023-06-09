# MapIR Band Correction

from MapIR import MapIR_RAW as mraw

import os



#------------------------------------------------
# Correction Process RAW:
#------------------------------------------------
file = 'RGN Files/AC 818/3.RAW'
image = mraw(file)
# image.display(hist=True)
image.NDVI(display=True, save=False)









