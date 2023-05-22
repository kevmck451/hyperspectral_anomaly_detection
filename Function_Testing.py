# Program to Test Each Function in Hyperspectral 1.2
#Kevin McKenzie 2022

import matplotlib.pyplot as plt
import numpy as np
import random
import os

from Hyperspectral_1_2 import Hyperspectral as h
import Hyperspectral_1_2 as hs
from Hyperspectral_1_2 import time_class as t

file = hs.AZ_22_Pro[3]

process1 = t('HS Object Creation')
image = h(file, stats=False)
process1.stats()


process2 = t('Image Info Functions')
image.pixel_metadata(x=200, y=200)
image.area_metadata(y1=10, y2=20, x1=10, x2=20)
image.image_metadata()
process2.stats()


process3 = t('Preprocessing Functions')
x1, x2, y1, y2 = 20, 100, 20, 100
bounds = [x1, x2, y1, y2]
image = image.crop(bounds)
image.graph_spectra_all_pixels()
image.preprocess(bands=(410, 900))
process3.stats()