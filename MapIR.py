#Analysis of Data from MapIR Camera
#Kevin McKenzie 2022

import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as iio
import os
import math


class MapIR:

    def __init__(self, raw_file_path):

        self.file_path = raw_file_path
        self.data = iio.imread(self.file_path)
        # print(self.data.shape) #5971 7406 4

        self.img_y = self.data.shape[0]
        self.img_x = self.data.shape[1]
        self.img_bands = self.data.shape[2]

        self.g_band = 550
        self.r_band = 660
        self.ir_band = 850

        self.R_index = 0
        self.G_index = 1
        self.NIR_index = 2

        #----------- This part is specifically for MapIR MonoChrom Test

        n = self.file_path
        n = n.split('/')
        try:
            n = n[2]
            n = n.split('.')
        except:
            pass

        self.band = n[0]
        # print(self.band)

    # Function to display the NRG Image
    def display_image(self):
        plt.imshow(self.data)
        plt.show()

    # Function to display the Normalized Difference Veg Index
    def NDVI(self):
        NIR = self.data[:, :, self.NIR_index] #2
        RED = self.data[:, :, self.R_index] #1
        RED, NIR = RED.astype('float32'), NIR.astype('float32')
        RED[RED == 0], NIR[NIR == 0] = np.nan, np.nan

        # calculate ndvi
        ndvi_array = (NIR - RED) / (NIR + RED)

        # plot. all of this is matplotlib ---------->
        ndvi_array_temp = np.zeros(ndvi_array.shape, dtype=float)
        ndvi_array_temp[ndvi_array >= 0.1] = ndvi_array[ndvi_array >= 0.1]

        # array_max = np.amax(ndvi_array_temp)
        # print(array_max)

        plt.figure(figsize=(18, 8))
        plt.imshow(ndvi_array_temp, cmap=plt.get_cmap("RdYlGn"))
        # plt.imshow(ndvi_array_temp, vmax= 0.55, cmap=plt.get_cmap("RdYlGn"))
        plt.title('NDVI')
        plt.colorbar()
        plt.axis('off')
        plt.show()

    # Function to display the Normalized Difference Veg Index
    def MSAVI(self):
        NIR = self.data[:, :, self.NIR_index] #2
        RED = self.data[:, :, self.R_index] #1
        RED, NIR = RED.astype('float32'), NIR.astype('float32')
        RED[RED == 0], NIR[NIR == 0] = np.nan, np.nan

        # calculate msavi
        aa = (2 * NIR + 1)**2
        ab = aa - 8 * (NIR - RED)
        bb = ab**0.5
        cc = 2 * NIR + 1 - bb
        msavi_array = cc / 2

        # plot. all of this is matplotlib ---------->
        msavi_array_temp = np.zeros(msavi_array.shape, dtype=float)
        msavi_array_temp[msavi_array >= 0.1] = msavi_array[msavi_array >= 0.1]

        # array_max = np.amax(msavi_array_temp)
        # print(array_max)

        plt.figure(figsize=(18, 8))
        # plt.imshow(msavi_array_temp, cmap=plt.get_cmap("RdYlGn"))
        plt.imshow(msavi_array_temp, vmax= 0.75, cmap=plt.get_cmap("RdYlGn"))
        plt.title('MSAVI')
        plt.colorbar()
        plt.axis('off')
        plt.show()

    # Function to display the Soil Adjusted Veg Index
    def SAVI(self, L):

        NIR = self.data[:, :, self.NIR_index]
        RED = self.data[:, :, self.R_index]
        RED, NIR = RED.astype('float32'), NIR.astype('float32')
        RED[RED == 0], NIR[NIR == 0] = np.nan, np.nan

        # calculate savi
        ndvi_array = ((NIR - RED) / (NIR + RED + L)) * (1 + L)

        # plot. all of this is matplotlib ---------->
        # ndvi_array_temp = np.zeros(ndvi_array.shape, dtype=float)
        # ndvi_array_temp[ndvi_array >= 0.1] = ndvi_array[ndvi_array >= 0.1]

        plt.figure(figsize=(18, 8))
        plt.imshow(ndvi_array, cmap=plt.get_cmap("RdYlGn"))
        plt.title('SAVI')
        plt.colorbar()
        plt.axis('off')
        plt.show()

    # Function to display the Enhanced Veg Index
    def EVI_2(self):

        NIR = self.data[:, :, self.NIR_index]
        RED = self.data[:, :, self.R_index]
        RED, NIR = RED.astype('float32'), NIR.astype('float32')
        RED[RED == 0], NIR[NIR == 0] = np.nan, np.nan

        # calculate EVI_2
        ndvi_array = 2.5 * ( NIR - RED) / ( NIR + 2.4 * RED + 1.0 )

        # plot. all of this is matplotlib ---------->
        # ndvi_array_temp = np.zeros(ndvi_array.shape, dtype=float)
        # ndvi_array_temp[ndvi_array >= 0.1] = ndvi_array[ndvi_array >= 0.1]

        plt.figure(figsize=(18, 8))
        plt.imshow(ndvi_array, cmap=plt.get_cmap("RdYlGn"))
        plt.title('EVI 2')
        plt.colorbar()
        plt.axis('off')
        plt.show()

    # Function to create a material from single pixel in image
    def graph_spectra_pixel(self, location, title, single):
        values_single = []
        for i in range(self.img_bands):
            values_single.append(self.data[location[1], location[0], i])

        bands = [self.g_band, self.r_band, self.ir_band]
        plt.plot(bands, values_single, linewidth=2, label = title)
        plt.xlabel('Bands')
        plt.ylabel('Values')
        plt.title('New Material', fontsize=20)
        plt.legend(loc='upper right')
        if single:
            plt.show()


class MapIRChromTest:

    def __init__(self):

        directory = 'RGN Files/MonoChrom Test'
        self.image_list_all = []
        self.band_list = []

        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            if os.path.isfile(f):
                im = MapIR(f)
                self.image_list_all.append(im)
                self.band_list.append(int(im.band))

        self.band_list.sort()
        self.red_list = []
        self.green_list = []
        self.nir_list = []

    #Function to create list of values from single pixel
    def get_values_pixel(self, pixel):
        for im in self.image_list_all:
            self.red_list.append( [ int(im.band), im.data[pixel[1], pixel[0], im.R_index] ] )  # 0
            self.green_list.append( [ int(im.band), im.data[pixel[1], pixel[0], im.G_index] ] )  # 1
            self.nir_list.append( [ int(im.band), im.data[pixel[1], pixel[0], im.NIR_index] ] ) # 2

        self.red_list.sort()
        self.green_list.sort()
        self.nir_list.sort()

        self.red_values = []
        self.green_values = []
        self.nir_values = []

        for i in range(len(self.red_list)):
            self.red_values.append(self.red_list[i][1])
            self.green_values.append(self.green_list[i][1])
            self.nir_values.append(self.nir_list[i][1])

    #Function to create list of values from single pixel
    def get_values_area(self, area):

        # another for loop to go through all the pixel coordinates
        red_group_list = []
        green_group_list = []
        nir_group_list = []

        red_av = []
        green_av = []
        nir_av = []

        for pixel in area:

            self.red_list = []
            self.green_list = []
            self.nir_list = []

            for im in self.image_list_all:
                self.red_list.append( [ int(im.band), im.data[pixel[1], pixel[0], im.R_index] ] )  # 0
                self.green_list.append( [ int(im.band), im.data[pixel[1], pixel[0], im.G_index] ] )  # 1
                self.nir_list.append( [ int(im.band), im.data[pixel[1], pixel[0], im.NIR_index] ] ) # 2

            self.red_list.sort()
            self.green_list.sort()
            self.nir_list.sort()

            self.red_values = []
            self.green_values = []
            self.nir_values = []

            for i in range(len(self.red_list)):
                self.red_values.append(self.red_list[i][1])
                self.green_values.append(self.green_list[i][1])
                self.nir_values.append(self.nir_list[i][1])

            red_group_list.append(self.red_values)
            green_group_list.append(self.green_values)
            nir_group_list.append(self.nir_values)

        for i in range(len(self.red_values)):
            rv = []
            gv = []
            nv = []

            for j in range(len(red_group_list)):
                rv.append(red_group_list[j][i])
                gv.append(green_group_list[j][i])
                nv.append(nir_group_list[j][i])

            red_av.append(np.mean(rv))
            green_av.append(np.mean(gv))
            nir_av.append(np.mean(nv))

        self.red_values = red_av
        self.green_values = green_av
        self.nir_values = nir_av

    #Function to graph each values in each band
    def graph(self):

        plt.plot(self.band_list, self.red_values, color='r', linewidth=2, label='Red Values')
        plt.plot(self.band_list, self.green_values, color='g', linewidth=2, label='Green Values')
        plt.plot(self.band_list, self.nir_values, color='b', linewidth=2, label='NIR Values')
        plt.vlines(x=[550, 650, 850], ymin=0, ymax=255, colors='black', ls='--', lw=1,
                   label='MapIR Bands')
        plt.xlabel('Bands')
        plt.ylabel('Values')
        plt.title('MapIR Monochromator Test 1', fontsize=20)
        plt.legend(loc='upper left')
        plt.show()
