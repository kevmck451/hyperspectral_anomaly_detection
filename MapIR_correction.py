# MapIR Band Correction

from MapIR import MapIRChromTest as mct
from MapIR import MapIRChromTestRAW as mctraw
from Hyperspectral_1_2 import Hyperspectral as hs
import Hyperspectral_1_2 as hype
from MapIR import MapIR_RAW as mraw
import MapIR as mir
import os

#------------------------------------------------
# ---------- MONO TEST 4 with RAW ---------------
#------------------------------------------------
pixel_sample = (1837, 1330)
wavelength_adjust_list = hype.wavelength_correct()
directory = 'Datasets/RGN Files/MonoChrom Test 4/raw'
test4 = mctraw(directory, wavelength_adjust_list, corr=True)
test4.get_values_area(pixel_sample)
test4.graph()
# correction_matrix = test4.integrate_np(display=True, stats=False, prnt=True)

# image_matrix = [[336.68, 74.61, 37.63], [33.52, 347.5, 41.77], [275.41, 261.99, 286.5]]
# image_matrix = [[7.18, 2.12, 1.02], [0.72, 9.86, 1.12], [5.88, 7.43, 7.74]]
# image_matrix = [[336, 33, 275], [74, 347, 261], [37, 41, 286]]

#------------------------------------------------
# Correction Process:
#------------------------------------------------
# file = 'RGN Files/AC 818/1.RAW'
# file = 'RGN Files/MonoChrom Test 4/raw/850.RAW'
# image = mraw(file)
# image.correction(stats=False, save=False)
# image.display_corrected()
# image.NDVI(threshold=False, display=True, save=False)

#------------------------------------------------
# Pika vs MapIR Comparison:
#------------------------------------------------
# pika_file = 'Datasets/PikaMapIR/az_cube_2 Processed.bip'
# image_p = hs(pika_file)
# image_p.display_RGB(display=True, save=False)
# image_p.display_Mapir_Single(display=True, save=False)
# image_p.display_NDVI(display=True, save=False)
#
# mapir_file = 'Datasets/PikaMapIR/MapIR_AZ.RAW'
# image_m = mraw(mapir_file)
# # image_m.display(hist=True)
# image_m.correction(stats=False, save=False)
# # image_m.display_corrected()
# image_m.NDVI(corr=True, display=True, save=False)




#------------------------------------------------
# Pika vs MapIR NDVI Comparison:
#------------------------------------------------
# pika_middle_tree = (141,304)
# pika_middle_fence = (134,476)
# pika_file = 'Datasets/PikaMapIR/az_cube_2 Processed.bip'
# image_p = hs(pika_file)
# # image_p.display_NDVI(display=True, save=False)
# image_p.NDVI_area_values(pika_middle_fence)


# mapir_middle_tree = (2480,756)
# mapir_middle_fence = (2382,1515)
# mapir_file = 'Datasets/PikaMapIR/MapIR_AZ.RAW'
# image_m = mraw(mapir_file)
# image_m.correction(stats=False, save=False)
# # image_m.NDVI(corr=True, display=True, save=False)
# image_m.NDVI_area_values(corr=False, middle_pixel=mapir_middle_fence)







