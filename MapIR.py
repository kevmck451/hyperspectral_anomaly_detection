#Analysis of Data from MapIR Camera
#Kevin McKenzie 2022

import matplotlib.pyplot as plt
import matplotlib.image as pltim
from copy import deepcopy
import imageio.v3 as iio
from PIL import Image
import numpy as np
import math
import time
import os



save_to_location = '../2 MapIR/Autosaves'

# RAW DATA CONVERSION CLASS
class MapIR_RAW:
    def __init__(self, raw_file_path):
        self.debay = False
        self.unp = False
        self.file_path = raw_file_path
        self.corrected = False

        # Get File Type
        try:
            file = raw_file_path.split('.')
            self.file_type = file[1]

            try:
                file_name = file[0].split('/')
                self.file_name = file_name[-1]
                # print(file_name[-1])
            except:
                self.file_name = file[0]
        except:
            print()
            # print('No File Type')

        with open(raw_file_path, "rb") as f:
            self.data = f.read()
            # print(self.data)

        # ----------------- UNPACK RAW DATA --------------------
        # Function to unpack MapIR raw image file per-pixel 12 bit values
        try:
            self.unp = True
            # unpack a mapir raw image file into per-pixel 12 bit values
            # print(len(self.data))
            assert len(self.data) == 1.5 * 4000 * 3000

            # two pixels are packed into each 3 byte value
            # ABC = nibbles of even pixel (big endian)
            # DEF = nibbles of odd pixel (big endian)
            # bytes in data: BC FA DE

            self.data = np.frombuffer(self.data, dtype=np.uint8).astype(np.uint16)

            # pull the first of every third byte as the even pixel data
            even_pixels = self.data[0::3]
            # pull the last of every third byte as the odd pixel data
            odd_pixels = self.data[2::3]
            # the middle byte has bits of both pixels
            middle_even = self.data[1::3].copy()
            middle_even &= 0xF
            middle_even <<= 8

            middle_odd = self.data[1::3].copy()
            middle_odd &= 0xF0
            middle_odd >>= 4
            odd_pixels <<= 4

            # combine middle byte data into pixel data
            even_pixels |= middle_even
            odd_pixels |= middle_odd

            pixels = np.stack((even_pixels, odd_pixels), axis=-1)

            # reshape to form camera image 4000x3000 pixels
            image = pixels.reshape((3000, 4000))
            self.data = image
            # print(self.data.shape)

            self.img_y, self.img_x, self.img_bands = self.data.shape[0], self.data.shape[1], 3
            self.g_band, self.r_band, self.ir_band = 550, 660, 850
            self.R_index, self.G_index, self.NIR_index = 0, 1, 2

            # ----------- This part is specifically for MapIR MonoChrom Test
            n = self.file_path
            n = n.split('/')
            try:
                n = n[-1]
                n = n.split('.')
            except:
                pass

            self.band = n[0]
            # print(self.band)

            self.debayer()
            # self.correct()

            # self.white_balance()
            # self.render_RGB()

        except:
            print()
            # print('File Corrupt')

    # Function to Debayer image matrix
    def debayer(self):
        import cv2
        # the bayer pattern is
        # R G
        # G B

        # use opencv's debayering routine on the data
        # COLOR_BAYER_RG2RGB # Company
        # COLOR_BAYER_BG2RGB # Thomas
        debayered_data = cv2.cvtColor(self.data, cv2.COLOR_BAYER_BG2RGB)
        # print(self.data[1330:1332, 1836:1838])

        output_data = (debayered_data >> 4).astype(np.uint8)

        # print(output_data.mean(axis=(0, 1)))

        self.data = output_data

    # Function to Correct White Balance
    def white_balance(self):
        self.data = self.data / 65535.0

        # Reapplies true color values to the pixel data if the -wb flag is set
        def color_correction(color):
            COLOR_CORRECTION_VECTORS = [1.398822546, -0.09047482163, 0.1619316638, -0.01290435996, 0.8994362354,
                                        0.1134681329,
                                        0.007306902204, -0.05995989591, 1.577814579]  # 101018
            roff = 0.0
            goff = 0.0
            boff = 0.0

            red_coeffs = COLOR_CORRECTION_VECTORS[6:9]
            green_coeffs = COLOR_CORRECTION_VECTORS[3:6]
            blue_coeffs = COLOR_CORRECTION_VECTORS[:3]

            color[:, :, 2] = (red_coeffs[0] * color[:, :, 0]) + (red_coeffs[1] * color[:, :, 1]) + (
                    red_coeffs[2] * color[:, :, 2]) + roff
            color[:, :, 1] = (green_coeffs[0] * color[:, :, 0]) + (green_coeffs[1] * color[:, :, 1]) + (
                    green_coeffs[2] * color[:, :, 2]) + goff
            color[:, :, 0] = (blue_coeffs[0] * color[:, :, 0]) + (blue_coeffs[1] * color[:, :, 1]) + (
                    blue_coeffs[2] * color[:, :, 2]) + boff

            color[color > 1.0] = 1.0
            color[color < 0.0] = 0.0

            return color

        self.data = color_correction(self.data)

        self.data = self.data * 65535.0

        self.data = self.data.astype("uint32")
        self.data = self.data.astype("uint16")
        self.data[self.data >= 65535] = 65535

    # Function to display the data
    def render_RGB(self):
        Ra = self.data[:, :, 0]
        Ga = self.data[:, :, 1]
        Ba = self.data[:, :, 2]

        # get 8bit arrays for each band
        scale8bit = lambda a: ((a - a.min()) * (1 / (a.max() - a.min()) * 255)).astype('uint8')
        Ra8, Ga8, Ba8 = scale8bit(Ra), scale8bit(Ga), scale8bit(Ba)

        # set rescaled fill pixels back to 0 for each array
        Ra8[Ra == 0], Ga8[Ga == 0], Ba8[Ba == 0] = 0, 0, 0

        # make rgb stack
        rgb_stack = np.zeros((self.img_y, self.img_x, 3), 'uint8')
        rgb_stack[..., 0], rgb_stack[..., 1], rgb_stack[..., 2] = Ra8, Ga8, Ba8

        self.rgb_render = rgb_stack

    # Function to calibrate MAPIR Camera for analysis
    def correction(self, stats, save):
        self.corrected = True
        # image_matrix = [[336.68, 74.61, 37.63], [33.52, 347.5, 41.77], [275.41, 261.99, 286.5]]
        # image_matrix = [[7.18, 2.12, 1.02], [0.72, 9.86, 1.12], [5.88, 7.43, 7.74]]
        image_matrix = [[336, 33, 275], [74, 347, 261], [37, 41, 286]]

        # Calculate the inverse of the image matrix
        image_matrix = np.asarray(image_matrix)
        inverse_matrix = np.linalg.inv(image_matrix)
        if stats: print(inverse_matrix)

        # Multiply each value in each band by the corresponding value in the inverse matrix
        corrected_data = np.zeros(self.data.shape)
        for i in range(self.data.shape[0]):
            corrected_data[i] = (inverse_matrix @ self.data[i].T).T

        self.data_corrected = corrected_data

        if stats: print(f'Max: {self.data_corrected.max()}\nMin: {self.data_corrected.min()}')

        if save:
            name = self.file_name + '-corrected'
            save_to = self.file_path.split('.')
            save_to = save_to[0] + f'-{name}.png'
            im = Image.fromarray(self.data)
            im.save(save_to)

        image_copy = deepcopy(self)
        image_copy.data = corrected_data

        return image_copy

    # Function to calibrate MAPIR Camera for analysis
    def correct(self):
        self.corrected = True
        # image_matrix = [[336.68, 74.61, 37.63], [33.52, 347.5, 41.77], [275.41, 261.99, 286.5]]
        # image_matrix = [[7.18, 2.12, 1.02], [0.72, 9.86, 1.12], [5.88, 7.43, 7.74]]
        image_matrix = [[336, 33, 275], [74, 347, 261], [37, 41, 286]]

        # Calculate the inverse of the image matrix
        image_matrix = np.asarray(image_matrix)
        inverse_matrix = np.linalg.inv(image_matrix)

        # Multiply each value in each band by the corresponding value in the inverse matrix
        corrected_data = np.zeros(self.data.shape)
        for i in range(self.data.shape[0]):
            corrected_data[i] = (inverse_matrix @ self.data[i].T).T

        self.data = corrected_data

    # Function to display the data
    def display(self, hist):

        Ra = self.data[:, :, 0]
        Ga = self.data[:, :, 1]
        Ba = self.data[:, :, 2]

        # get 8bit arrays for each band
        scale8bit = lambda a: ((a - a.min()) * (1 / (a.max() - a.min()) * 255)).astype('uint8')
        Ra8, Ga8, Ba8 = scale8bit(Ra), scale8bit(Ga), scale8bit(Ba)

        # set rescaled fill pixels back to 0 for each array
        Ra8[Ra == 0], Ga8[Ga == 0], Ba8[Ba == 0] = 0, 0, 0

        # make rgb stack
        rgb_stack = np.zeros((self.img_y, self.img_x, 3), 'uint8')
        rgb_stack[..., 0], rgb_stack[..., 1], rgb_stack[..., 2] = Ra8, Ga8, Ba8

        # apply histogram equalization to each band
        if hist:
            for i in range(rgb_stack.shape[2]):
                # band i
                b = rgb_stack[:, :, i]
                # histogram from flattened (1d) image
                b_histogram, bins = np.histogram(b.flatten(), 256)
                # cumulative distribution function
                b_cumdistfunc = b_histogram.cumsum()
                # normalize
                b_cumdistfunc = 255 * b_cumdistfunc / b_cumdistfunc[-1]
                # get new values by linear interpolation of cdf
                b_equalized = np.interp(b.flatten(), bins[:-1], b_cumdistfunc)
                # reshape to 2d and add back to rgb_stack
                rgb_stack[:, :, i] = b_equalized.reshape(b.shape)

        self.rgb_graph = rgb_stack

        plt.imshow(rgb_stack, cmap=plt.get_cmap(None))
        plt.axis('off')
        plt.show()

    # Function to display the data
    def display_corrected(self):
        correction_plot = self.data_corrected.copy()
        correction_plot[correction_plot < 0] = 0
        correction_plot -= correction_plot.min()
        correction_plot /= 8

        # Display the first image
        plt.subplot(1, 2, 1)
        plt.imshow(self.data.copy(), cmap=plt.get_cmap(None))
        plt.axis('off')
        plt.title('Orignal Image')

        # Display the second image
        plt.subplot(1, 2, 2)
        plt.imshow(correction_plot, cmap=plt.get_cmap(None))
        plt.axis('off')
        plt.title('Corrected Image')

        # plt.subplot(1, 3, 3)
        # plt.hist(self.data_corrected.ravel(), bins=100)
        # plt.title('Corrected Image')

        # Show the plot
        plt.show()

    # Function to graph all spectral signature for every pixel in image
    def graph_mapir_pika(self, display, save):

        x_list = [i for i in range(int(self.img_x*.45), int(self.img_x*.55))]
        y_list = [i for i in range(int(self.img_y*.45), int(self.img_y*.55))]

        for i in x_list:
            for j in y_list:
                values_single = [self.data[int(j), int(i), k] for k in range(self.img_bands)]
                plt.plot([0,1,2], values_single[0:3], linewidth=2)
                plt.xlabel('Bands')
                plt.ylabel('Values')

        if save:
            saveas = f'../../Dropbox/2 Work/1 Optics Lab/2 Projects/MapIR/Autosaves/{self.file_path}-Graph'
            plt.savefig(saveas)
            plt.close()
        if display:
            plt.show()

    # Function to export processed data
    def export(self, directory):
        save_name = f'{directory}/{self.file_name}_corrected.RAW'

        # Make folder inside directory for corrected files

    # Function to display the Normalized Difference Veg Index
    def NDVI(self, display, save):

        NIR = self.data[:, :, self.NIR_index]
        RED = self.data[:, :, self.R_index]

        RED, NIR = RED.astype('float'), NIR.astype('float')
        # RED[RED == 0], NIR[NIR == 0] = np.nan, np.nan
        top, bottom = NIR - RED, NIR + RED
        top[top == 0], bottom[bottom == 0] = 0, np.nan

        ndvi_array = np.divide(top, bottom)
        ndvi_array[ndvi_array < 0] = 0
        ndvi_array[ndvi_array > 1] = 1

        plt.imshow(ndvi_array, cmap=plt.get_cmap("RdYlGn"))
        plt.title(f'NDVI')
        plt.colorbar()
        plt.axis('off')

        if display:
            plt.show()

        if save:
            saveas = (f'{save_to_location}/{self.file_name} NDVI-RAW')
            plt.savefig(saveas)
            plt.close()

    # Function to tell average values in an area
    def NDVI_area_values(self, corr, middle_pixel):

        if corr:
            name = 'Corrected'
            NIR = self.data_corrected[:, :, self.NIR_index]
            RED = self.data_corrected[:, :, self.R_index]

        else:
            name = 'Original'
            NIR = self.data[:, :, self.NIR_index]
            RED = self.data[:, :, self.R_index]

        RED, NIR = RED.astype('float'), NIR.astype('float')
        # RED[RED == 0], NIR[NIR == 0] = np.nan, np.nan
        top, bottom = NIR - RED, NIR + RED
        top[top == 0], bottom[bottom == 0] = 0, np.nan

        ndvi_array = np.divide(top, bottom)
        # ndvi_array[ndvi_array < 0] = 0
        # ndvi_array[ndvi_array > 1] = 1

        plus_minus = 100
        x1 = (middle_pixel[0] - plus_minus)
        x2 = (middle_pixel[0] + plus_minus)
        y1 = (middle_pixel[1] - plus_minus)
        y2 = (middle_pixel[1] + plus_minus)

        average_value = ndvi_array[y1:y2, x1:x2].mean()
        print(average_value)

    # Function to display the Normalized Difference Veg Index with Min/Max Specified
    def NDVI_Mm(self, display, save, mi, ma):
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

        plt.imshow(ndvi_array, vmin= mi, vmax= ma, cmap=plt.get_cmap("RdYlGn"))
        plt.title('NDVI')
        plt.colorbar()
        plt.axis('off')


        if display:
            plt.show()

        if save:
            saveas = (f'{save_to_location}/{self.file_name} NDVI')
            plt.savefig(saveas)
            plt.close()

# JPG DATA CONVERSION CLASS
class MapIR_jpg:
    def __init__(self, raw_file_path):

        self.file_path = raw_file_path

        #Get File Type
        try:
            file = raw_file_path.split('.')
            self.file_type = file[1]

            try:
                file_name = file[0].split('/')
                self.file_name = file_name[-1]
                # print(file_name[-1])
            except:
                self.file_name = file[0]
        except: print('No File Type')

        self.data = iio.imread(self.file_path)
        self.data = self.data[:, :, 0:3]
        # print(self.data.shape) #5971 7406 4

        self.img_y, self.img_x, self.img_bands = self.data.shape[0], self.data.shape[1], 3
        self.g_band, self.r_band, self.ir_band = 550, 660, 850
        self.R_index, self.G_index, self.NIR_index = 0, 1, 2

        #----------- This part is specifically for MapIR MonoChrom Test

        n = self.file_path
        n = n.split('/')
        try:
            n = n[-1]
            n = n.split('.')
        except:
            pass

        self.band = n[0]
        # print(self.band)

    # Function to display the RGN Image
    def display_image(self):
        plt.imshow(self.data)
        plt.axis('off')
        plt.show()

    # Function to display the Normalized Difference Veg Index with Min/Max Specified
    def NDVI_Mm(self, display, save, mi, ma):
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

        plt.imshow(ndvi_array_temp, vmin= mi, vmax= ma, cmap=plt.get_cmap("RdYlGn"))
        plt.title('NDVI')
        plt.colorbar()
        plt.axis('off')


        if display:
            plt.show()

        if save:
            saveas = (f'{save_to_location}/{self.file_name} NDVI')
            plt.savefig(saveas)
            plt.close()

    # Function to display the Normalized Difference Veg Index
    def NDVI(self, display, save):

        NIR = self.data[:, :, self.NIR_index]
        RED = self.data[:, :, self.R_index]

        RED, NIR = RED.astype('float'), NIR.astype('float')
        # RED[RED == 0], NIR[NIR == 0] = np.nan, np.nan
        top, bottom = NIR - RED, NIR + RED
        top[top == 0], bottom[bottom == 0] = 0, np.nan

        ndvi_array = np.divide(top, bottom)
        # ndvi_array[ndvi_array < 0] = 0
        # ndvi_array[ndvi_array > 1] = 1

        plt.imshow(ndvi_array, cmap=plt.get_cmap("RdYlGn"))
        plt.title(f'NDVI')
        plt.colorbar()
        plt.axis('off')

        if display:
            plt.show()

        if save:
            saveas = (f'{save_to_location}/{self.file_name} NDVI-RAW')
            plt.savefig(saveas)
            plt.close()

    # Function to display the Soil Adjusted Veg Index
    def SAVI(self, L):

        NIR = self.data[:, :, self.NIR_index]
        RED = self.data[:, :, self.R_index]
        RED, NIR = RED.astype('float32'), NIR.astype('float32')
        RED[RED == 0], NIR[NIR == 0] = np.nan, np.nan

        # calculate savi
        savi_array = ((NIR - RED) / (NIR + RED + L)) * (1 + L)

        # plot. all of this is matplotlib ---------->
        # ndvi_array_temp = np.zeros(ndvi_array.shape, dtype=float)
        # ndvi_array_temp[ndvi_array >= 0.1] = ndvi_array[ndvi_array >= 0.1]

        plt.figure(figsize=(18, 8))
        plt.imshow(savi_array, cmap=plt.get_cmap("RdYlGn"))
        plt.title('SAVI')
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
        plt.imshow(msavi_array, cmap=plt.get_cmap("RdYlGn"))
        # plt.imshow(msavi_array_temp, vmax= 0.75, cmap=plt.get_cmap("RdYlGn"))
        plt.title('MSAVI')
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

        values_single = [ self.data[location[1], location[0], i] for i in range(self.img_bands) ]

        bands = [self.g_band, self.r_band, self.ir_band]
        plt.plot(bands, values_single, linewidth=2, label = title)
        plt.xlabel('Bands')
        plt.ylabel('Values')
        plt.title('New Material', fontsize=20)
        # plt.legend(loc='upper right')
        if single:
            plt.show()

    # Function to graph all spectral signature for every pixel in image
    def graph_mapir_pika(self, display, save):

        x_list = np.linspace(0, (self.img_x - 1))
        y_list = np.linspace(0, (self.img_y - 1))

        for i in x_list:
            for j in y_list:
                values_single = [self.data[int(j), int(i), k] for k in range(self.img_bands)]
                plt.plot([0,1,2], values_single[0:3], linewidth=2)
                plt.xlabel('Bands')
                plt.ylabel('Values')

        if save:
            saveas = f'../../Dropbox/2 Work/1 Optics Lab/2 Projects/MapIR/Autosaves/{self.file_path}-Graph'
            plt.savefig(saveas)
            plt.close()
        if display:
            plt.show()

    # Function to calibrate MAPIR Camera for analysis
    def correct(self):
        self.corrected = True
        # image_matrix = [[336.68, 74.61, 37.63], [33.52, 347.5, 41.77], [275.41, 261.99, 286.5]]
        # image_matrix = [[7.18, 2.12, 1.02], [0.72, 9.86, 1.12], [5.88, 7.43, 7.74]]
        image_matrix = [[1201, 11, 1135], [13, 991, 477], [340, 12, 1721]]

        # Calculate the inverse of the image matrix
        image_matrix = np.asarray(image_matrix)
        inverse_matrix = np.linalg.inv(image_matrix)

        # Multiply each value in each band by the corresponding value in the inverse matrix
        corrected_data = np.zeros(self.data.shape)
        for i in range(self.data.shape[0]):
            corrected_data[i] = (inverse_matrix @ self.data[i].T).T

        self.data = corrected_data

    # Function to calibrate MAPIR Camera for analysis
    def correction(self, stats, save):
        self.corrected = True
        # image_matrix = [[336.68, 74.61, 37.63], [33.52, 347.5, 41.77], [275.41, 261.99, 286.5]]
        # image_matrix = [[7.18, 2.12, 1.02], [0.72, 9.86, 1.12], [5.88, 7.43, 7.74]]
        image_matrix = [[336, 33, 275], [74, 347, 261], [37, 41, 286]]

        # Calculate the inverse of the image matrix
        image_matrix = np.asarray(image_matrix)
        inverse_matrix = np.linalg.inv(image_matrix)
        if stats: print(inverse_matrix)

        # Multiply each value in each band by the corresponding value in the inverse matrix
        corrected_data = np.zeros(self.data.shape)
        for i in range(self.data.shape[0]):
            corrected_data[i] = (inverse_matrix @ self.data[i].T).T

        self.data_corrected = corrected_data

        if stats: print(f'Max: {self.data_corrected.max()}\nMin: {self.data_corrected.min()}')

        if save:
            name = self.file_name + '-corrected'
            save_to = self.file_path.split('.')
            save_to = save_to[0] + f'-{name}.png'
            im = Image.fromarray(self.data)
            im.save(save_to)

        image_copy = deepcopy(self)
        image_copy.data = corrected_data

        return image_copy

    # Function to categorize pixels in image
    def categorize_1(self, var):
        total_start_time = time.time()
        bands = [0,1,2]
        variation = var
        iterations = 100

        cll = (variation / 100) - 1
        clh = (variation / 100) + 1
        x_list = [i for i in range(self.img_x)]
        y_list = [i for i in range(self.img_y)]
        # print(self.img_x, self.img_y)

        # if there are bands outside the max num bands, remove
        while np.max(bands) >= self.img_bands: bands.pop()

        # INITIATE ALL PIXEL OBJECTS AND SAMPLE VALUES AT POINTS
        print('Initializing Pixel Objects')
        # print(f'Estimated Time: {((len(x_list)*len(y_list)) / 400_500 )/60} mins')
        t3 = time.time()
        pixel_values = [[[self.data[j, i, k] for k in bands] for j in y_list] for i in x_list]
        self.pixel_master_list = []
        for i in x_list:
            for j in y_list:
                p = pixel_class([i, j], np.asarray(pixel_values[i][j]))
                self.pixel_master_list.append(p)
        # print(f'Pixel Ob Time: {(time.time() - t3) / 60} mins')
        # print(f'Pix / s: {(len(x_list)*len(y_list) / (time.time() - t3))}')

        # CATEGORIZE ALL PIXELS BASED ON SIMILARITY
        def is_similar(compare_list, list):
            similar = [1] * len(compare_list)
            comp = []
            for x, y in zip(compare_list, list):
                if (x * cll) <= y <= (x * clh): comp.append(1)
                else: comp.append(0)
            # print('{} vs {}'.format(comp, similar))
            if comp == similar: return True
            else: return False

        def check_uncat():
            # print('Checking if any Uncategorized')
            for pixel in self.pixel_master_list:
                if pixel.subcat_num == 0:
                    # print('Found a 0')
                    return True
            # print('Everything is Categorized')
            return False

        # Categorizing ------------------------------------------
        print('Subcategorizing')
        self.subcat_master_list = []
        subcat_num = 1
        c = category_class()
        c.subcat_num = subcat_num
        self.subcat_master_list.append(c)
        f = True
        comp = []
        for pixel in self.pixel_master_list:
            if f:
                comp = pixel.values
                pixel.subcat_num = subcat_num
                c.pixel_list.append(pixel)
                pixel.subcategory = c
                f = False
                continue
            if is_similar(comp, pixel.values):
                # print('Similar')
                pixel.subcat_num = subcat_num
                c.pixel_list.append(pixel)
                pixel.subcategory = c

        category_class.max_subcategories = 1
        print('Subcat Num: {}'.format(subcat_num))

        # While Loop ------------------
        while check_uncat():
            subcat_num += 1
            c = category_class()
            c.subcat_num = subcat_num
            self.subcat_master_list.append(c)
            comp_w = []
            for pixel in self.pixel_master_list:
                if pixel.subcat_num == 0:
                    comp_w = pixel.values
                    pixel.subcat_num = subcat_num
                    c.pixel_list.append(pixel)
                    pixel.subcategory = c
                    break
            for pixel in self.pixel_master_list:
                if pixel.subcat_num == 0 and is_similar(comp_w, pixel.values):
                    pixel.subcat_num = subcat_num
                    c.pixel_list.append(pixel)
                    pixel.subcategory = c
            if subcat_num > iterations:
                for pixel in self.pixel_master_list:
                    if pixel.subcat_num == 0:
                        pixel.subcat_num = subcat_num
                        c.pixel_list.append(pixel)
                        pixel.subcategory = c
                        break
            category_class.max_subcategories += 1
            print('Subcat Num: {}'.format(subcat_num))
        # End of While Loop ---------------
        # print(f'Total Categories: {subcat_num}')
        # Everything is Categorized

        self.subcategory_data_dict = None

        def cat_tally_sort():
            subcat_list = [pixel.subcat_num for pixel in self.pixel_master_list]

            # Create dict of category numbers
            subcat_dict = {}
            for i in range(1, subcat_num + 1):
                # print(i)
                subcat_dict.update({i: subcat_list.count(i)})
            # print(cat_dict)
            # Sort dict in reverse order based on values
            subcat_dict = dict(sorted(subcat_dict.items(), key=lambda item: item[1], reverse=True))
            # print(cat_dict)

            new_subcat_num_dict = {}
            for i, x in zip(range(1, len(subcat_dict) + 1), subcat_dict.keys()):
                new_subcat_num_dict.update({x: i})

            # print(new_cat_num_dict)

            for pixel in self.pixel_master_list:
                # print(pixel.cat_num, new_cat_num_dict.get(pixel.category))
                if pixel.subcat_num != new_subcat_num_dict.get(pixel.subcat_num):
                    pixel.subcat_num = new_subcat_num_dict.get(pixel.subcat_num)

            subcat_list = [pixel.subcat_num for pixel in self.pixel_master_list]

            # Create dict of category numbers
            subcat_dict = {}
            for i in range(1, subcat_num + 1):
                subcat_dict.update({i: subcat_list.count(i)})

            # Sort dict in reverse order based on values
            self.subcategory_data_dict = dict(sorted(subcat_dict.items(), key=lambda item: item[1], reverse=True))
            print(self.subcategory_data_dict)

            for subcat in self.subcat_master_list:
                if subcat.subcat_num != new_subcat_num_dict.get(subcat.subcat_num):
                    subcat.subcat_num = new_subcat_num_dict.get(subcat.subcat_num)

        cat_tally_sort()

        print(f'Total Time: {(time.time() - total_start_time) / 60} mins')

    # Function to display all subcategories
    def display_categories(self, display, cutoff_percent=100):

        # Create Array with pixels xy with cat value
        cat_array = np.zeros((self.data.shape[0], self.data.shape[1], 1), dtype=float)
        cutoff = (len(self.subcategory_data_dict) + 1) * (cutoff_percent / 100)
        cutoff = int((len(self.subcategory_data_dict) + 1) - cutoff)
        # print(cutoff)

        for pixel in self.pixel_master_list:

            if pixel.subcat_num > cutoff:
                cat_array[pixel.location[1], pixel.location[0], 0] = pixel.subcat_num
            else:
                cat_array[pixel.location[1], pixel.location[0], 0] = 0

        # Display Image with Categorized Pixels
        # plt.figure(figsize=(10, 10))
        # og = plt.imshow(self.data[:, :, 50], cmap='gray')

        if display == True:
            plt.imshow(cat_array, cmap='nipy_spectral')
            plt.title('Category Overlay')
            plt.colorbar()
            plt.axis('off')
            plt.show()
        else:
            self.graph_subcat = cat_array

# MAPIR MONOCHROMATOR TEST CLASS FOR RAW
class MapIRChromTestRAW:
    def __init__(self, dir, adjust, corr):

        adjustment = adjust
        directory = dir
        self.image_list_all = []
        self.band_list = []

        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            if os.path.isfile(f):
                try:
                    im = MapIR_RAW(f)
                    if corr:
                        im = im.correction(stats=False, save=False)
                        self.image_list_all.append(im)
                    else:
                        self.image_list_all.append(im)
                    self.band_list.append(int(im.band))
                except:
                    continue

        self.band_list.sort()
        new_band_list = []
        if len(adjustment) > 0:
            for band, new_band in zip(self.band_list, adjustment):
                new_band_list.append(band + new_band)
        self.band_list = new_band_list
        self.red_list = []
        self.green_list = []
        self.nir_list = []

    # Function to create list of values from single pixel
    def get_values_pixel(self, pixel):
        for im in self.image_list_all:
            try:
                self.red_list.append( [ int(im.band), im.data[pixel[1], pixel[0], im.R_index] ] )  # 0
                self.green_list.append( [ int(im.band), im.data[pixel[1], pixel[0], im.G_index] ] )  # 1
                self.nir_list.append( [ int(im.band), im.data[pixel[1], pixel[0], im.NIR_index] ] ) # 2

            except:
                print(f'EXCEPTION: {im.file_name}')
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

    # Function to create list of values from single pixel
    def get_values_area(self, pixel):

        for im in self.image_list_all:
            try:
                y1 = pixel[1]-20
                y2 = pixel[1]+20
                x1 = pixel[0]-20
                x2 = pixel[0]+20
                self.red_list.append([int(im.band), im.data[y1:y2, x1:x2, im.R_index].mean()])  # 0
                self.green_list.append([int(im.band), im.data[y1:y2, x1:x2, im.G_index].mean()])  # 1
                self.nir_list.append([int(im.band), im.data[y1:y2, x1:x2, im.NIR_index].mean()])  # 2
            except:
                print(f'EXCEPTION: {im.file_name}')

        # print(self.red_list)

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

    # Function to graph each values in each band
    def graph(self):
        max = np.max(self.red_values)
        if not self.image_list_all[0].corrected:
            self.red_values.append(1)
            self.green_values.append(1)
            self.nir_values.append(1)
            self.red_values.append(0)
            self.green_values.append(0)
            self.nir_values.append(0)
            self.band_list.append(880)
            self.band_list.append(890)

        plt.plot(self.band_list, self.red_values, color='r', linewidth=2, label='Red Values')
        plt.plot(self.band_list, self.green_values, color='g', linewidth=2, label='Green Values')
        plt.plot(self.band_list, self.nir_values, color='b', linewidth=2, label='NIR Values')
        plt.vlines(x=[550, 660, 850], ymin=0, ymax=max, colors='black', ls='--', lw=1, label='MapIR Bands')
        plt.xlabel('Bands')
        plt.ylabel('Counts')
        plt.xticks([x for x in range(500, 900, 25)])
        plt.title('MapIR Monochromator Test: RAW', fontsize=20)
        plt.legend(loc='upper right')
        plt.show()

    # Function to integrate the response for each band for calibration using numpy
    def integrate_np(self, display, stats, prnt):

        # Integration variables
        ra1, ra2, ra3, ga1, ga2, ga3, na1, na2, na3 = 0, 0, 0, 0, 0, 0, 0, 0, 0

        # Integration values from numpy
        # 500-600nm
        yr1 = [0, *self.red_values[4:20], 0]
        yg1 = [0, *self.green_values[4:20], 0]
        yn1 = [0, *self.nir_values[6:20], 0]
        ga1 = np.trapz(yr1)
        ga2 = np.trapz(yg1)
        ga3 = np.trapz(yn1)

        # 600-700nm
        yr2 = [0, *self.red_values[25:40], 0]
        yg2 = [0, *self.green_values[25:40], 0]
        yn2 = [0, *self.nir_values[25:40], 0]
        ra1 = np.trapz(yr2)
        ra2 = np.trapz(yg2)
        ra3 = np.trapz(yn2)

        # 700-850nm
        yr3 = [0, *self.red_values[41:56], 0]
        yg3 = [0, *self.green_values[41:56], 0]
        yn3 = [0, *self.nir_values[41:56], 0]
        na1 = np.trapz(yr3)
        na2 = np.trapz(yg3)
        na3 = np.trapz(yn3)

        redn = ['Ra1', 'Ra2', 'Ra3']
        greenn = ['Ga1', 'Ga2', 'Ga3']
        nirn = ['Na1', 'Na2', 'Na3']
        # red = [int(ra1), int(ra2), int(ra3)]
        # green = [int(ga1), int(ga2), int(ga3)]
        # nir = [int(na1), int(na2), int(na3)]

        red = [round(ra1, 2), round(ra2, 2), round(ra3, 2)]
        green = [round(ga1, 2), round(ga2, 2), round(ga3, 2)]
        nir = [round(na1, 2), round(na2, 2), round(na3, 2)]

        if display:
            plt.bar(redn, red, color=['red', 'green', 'blue'])
            plt.bar(greenn, green, color=['red', 'green', 'blue'])
            plt.bar(nirn, nir, color=['red', 'green', 'blue'])
            plt.text(redn[0], red[0], red[0], ha='center', )
            plt.text(redn[1], red[1], red[1], ha='center')
            plt.text(redn[2], red[2], red[2], ha='center')
            plt.text(greenn[0], green[0], green[0], ha='center', )
            plt.text(greenn[1], green[1], green[1], ha='center')
            plt.text(greenn[2], green[2], green[2], ha='center')
            plt.text(nirn[0], nir[0], nir[0], ha='center', )
            plt.text(nirn[1], nir[1], nir[1], ha='center')
            plt.text(nirn[2], nir[2], nir[2], ha='center')
            plt.title(f'RGN Integration Values: RAW')
            plt.ylabel('Values')
            plt.show()

            # plt.barh(redn, red, color=['red', 'green', 'blue'])
            # plt.barh(greenn, green, color=['red', 'green', 'blue'])
            # plt.barh(nirn, nir, color=['red', 'green', 'blue'])
            # plt.text(red[0], redn[0], red[0], va='center')
            # plt.text(red[1], redn[1], red[1], va='center')
            # plt.text(red[2], redn[2], red[2], va='center')
            # plt.text(green[0], greenn[0], green[0], va='center')
            # plt.text(green[1], greenn[1], green[1], va='center')
            # plt.text(green[2], greenn[2], green[2], va='center')
            # plt.text(nir[0], nirn[0], nir[0], va='center')
            # plt.text(nir[1], nirn[1], nir[1], va='center')
            # plt.text(nir[2], nirn[2], nir[2], va='center')
            # plt.title(f'RGN Integration Values: RAW')
            # plt.xlabel('Values')
            # plt.show()

        # rsum = 3750 / 80
        # gsum = 2644 / 75
        # nsum = 2961 / 80

        # Ra1, Ra2, Ra3 = round((red[0] / rsum), 2), round((red[1] / gsum), 2), round((red[2] / nsum), 2)
        # Ga1, Ga2, Ga3 = round((green[0] / rsum), 2), round((green[1] / gsum), 2), round((green[2] / nsum), 2)
        # Na1, Na2, Na3 = round((nir[0] / rsum), 2), round((nir[1] / gsum), 2), round((nir[2] / nsum), 2)

        # Ra1, Ra2, Ra3 = round((red[0]), 2), round((red[1]), 2), round((red[2]), 2)
        # Ga1, Ga2, Ga3 = round((green[0]), 2), round((green[1]), 2), round((green[2]), 2)
        # Na1, Na2, Na3 = round((nir[0]), 2), round((nir[1]), 2), round((nir[2]), 2)

        Ra1, Ra2, Ra3 = round((red[0]), 2), round((green[0]), 2), round((nir[0]), 2)
        Ga1, Ga2, Ga3 = round((red[1]), 2), round((green[1]), 2), round((nir[1]), 2)
        Na1, Na2, Na3 = round((red[2]), 2), round((green[2]), 2), round((nir[2]), 2)

        calibration = [[Ra1, Ra2, Ra3], [Ga1, Ga2, Ga3], [Na1, Na2, Na3]]

        if stats:
            print(f'RED: {self.red_values}')
            print('-' * 30)
            print(f'GREEN: {self.green_values}')
            print('-' * 30)
            print(f'NIR: {self.nir_values}')
            print(f'Y-R: {yr1}')
            print(f'Y-G: {yg1}')
            print(f'Y-N: {yn1}')
            print(ga1, ga2, ga3)
            print(f'Y-R: {yr2}')
            print(f'Y-G: {yg2}')
            print(f'Y-N: {yn2}')
            print(ra1, ra2, ra3)
            print(f'Y-R: {yr3}')
            print(f'Y-G: {yg3}')
            print(f'Y-N: {yn3}')
            print(na1, na2, na3)
            print(calibration)

        if prnt:
            print(f'        [ R    G    N   ]')
            print(f'RED   = [{Ra1}, {Ra2}, {Ra3}]')
            print(f'GREEN = [{Ga1}, {Ga2}, {Ga3}]')
            print(f'NIR =   [{Na1}, {Na2}, {Na3}]')
            print(f'Calibration   = [[{Ra1}, {Ra2}, {Ra3}], [{Ga1}, {Ga2}, {Ga3}], [{Na1}, {Na2}, {Na3}]]')

        return calibration

    # Function to get the radiance calibration values
    def radiance_calibration(self):
        pass

# MAPIR MONOCHROMATOR TEST CLASS FOR JPGS
class MapIRChromTest:
    def __init__(self, dir, adjust):
        adjustment = adjust
        directory = dir
        self.image_list_all = []
        self.band_list = []

        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            if os.path.isfile(f):
                im = MapIR_jpg(f)
                self.image_list_all.append(im)
                self.band_list.append(int(im.band))

        self.band_list.sort()
        new_band_list = []
        if len(adjustment) > 0:
            for band, new_band in zip(self.band_list, adjustment):
                new_band_list.append(band + new_band)
        self.band_list = new_band_list
        self.red_list = []
        self.green_list = []
        self.nir_list = []

    # Function to create list of values from single pixel
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

    # Function to create list of values from sampled area of pixel
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

    # Function to graph each values in each band
    def graph(self):
        self.red_values.append(1)
        self.green_values.append(1)
        self.nir_values.append(1)
        self.red_values.append(0)
        self.green_values.append(0)
        self.nir_values.append(0)
        self.band_list.append(880)
        self.band_list.append(890)
        plt.plot(self.band_list, self.red_values, color='r', linewidth=2, label='Red Values')
        plt.plot(self.band_list, self.green_values, color='g', linewidth=2, label='Green Values')
        plt.plot(self.band_list, self.nir_values, color='b', linewidth=2, label='NIR Values')
        plt.vlines(x=[550, 650, 850], ymin=0, ymax=255, colors='black', ls='--', lw=1, label='MapIR Bands')
        plt.xlabel('Bands')
        plt.ylabel('Counts')
        plt.xticks([x for x in range(500, 900, 25)])
        plt.title('MapIR Monochromator Test-JPGS', fontsize=20)
        plt.legend(loc='upper left')
        plt.show()

    # Function to integrate the response for each band for calibration
    def integrate_1(self):
        green_total_int = 0
        red_total_int = 0
        nir_total_int = 0

        for g, r, n in zip(self.green_values, self.red_values, self.nir_values):
            green_total_int += g
            red_total_int += r
            nir_total_int += n

        print(f'{green_total_int} : {red_total_int} : {nir_total_int}')

        bands = ['G', 'R', 'N']
        values = [red_total_int, green_total_int, nir_total_int]

        plt.bar(bands, values, color = ['red', 'green', 'blue'])
        plt.text(bands[0], values[0], values[0], ha='center', )
        plt.text(bands[1], values[1], values[1], ha='center')
        plt.text(bands[2], values[2], values[2], ha='center')
        plt.title('RGN Integration Values: Int 1')
        plt.ylabel('Values')
        plt.show()

    # Function to integrate the response for each band for calibration
    def integrate_bw(self, samp, display):
        ga1 = 0 # actually 0
        ga2 = 0
        ga3 = 0 # actually 0
        ra1 = 0
        ra2 = 0 # actually 0
        ra3 = 0
        na1 = 0
        na2 = 0
        na3 = 0
        prev_g = 0
        prev_r = 0
        prev_n = 0
        mg = 0
        mr = 0
        mn = 0
        bg = 0
        br = 0
        bn = 0

        #10-19
        #20-29
        #36-46

        for i, (g, r, n) in enumerate(zip(self.green_values, self.red_values, self.nir_values)):
            # print(f'------{i}--------')
            if i == 0:
                continue
            g = int(g)
            r = int(r)
            n = int(n)

            if 10 <= i <= 19:
                mg = (g - prev_g) / (i - (i-1))
                # print(f'Slope: {mg} : {mr} : {mn}')
                bg = g - (mg * i)
                # print(f'Y-int: {bg} : {br} : {bn}')
                samples = np.linspace((i-1), i, samp)
                # print(f'Samples: {samples}')

                for s in samples:
                    dx = (samples[1]-samples[0])
                    ga2 += dx * (mg * s + bg)

                prev_g = g

            if 21 <= i <= 30:
                mr = (r - prev_r) / (i - (i-1))
                mn = (n - prev_n) / (i - (i - 1))
                # print(f'Slope: {mg} : {mr} : {mn}')
                br = r - (mr * i)
                bn = n - (mn * i)
                # print(f'Y-int: {bg} : {br} : {bn}')
                samples = np.linspace((i-1), i, samp)
                # print(f'Samples: {samples}')

                for s in samples:
                    dx = (samples[1]-samples[0])
                    ra1 += dx * (mr * s + br)
                    ra3 += dx * (mn * s + bn)

                prev_r = r
                prev_n = n

            if 37 <= i <= 47:
                mg = (g - prev_g) / (i - (i - 1))
                mr = (r - prev_r) / (i - (i-1))
                mn = (n - prev_n) / (i - (i - 1))
                # print(f'Slope: {mg} : {mr} : {mn}')
                bg = g - (mg * i)
                br = r - (mr * i)
                bn = n - (mn * i)
                # print(f'Y-int: {bg} : {br} : {bn}')
                samples = np.linspace((i-1), i, samp)
                # print(f'Samples: {samples}')

                for s in samples:
                    dx = (samples[1]-samples[0])
                    na1 += dx * (mr * s + br)
                    na2 += dx * (mg * s + bg)
                    na3 += dx * (mn * s + bn)

                prev_g = g
                prev_r = r
                prev_n = n

        redn = ['Ra1', 'Ra2', 'Ra3']
        greenn = ['Ga1', 'Ga2', 'Ga3']
        nirn = ['Na1', 'Na2', 'Na3']
        red = [int(ra1), int(ra2), int(ra3)]
        green = [int(ga1), int(ga2), int(ga3)]
        nir = [int(na1), int(na2), int(na3)]

        if display:
            plt.bar(greenn, green, color=['red', 'green', 'blue'])
            plt.bar(redn, red, color = ['red', 'green', 'blue'])
            plt.bar(nirn, nir, color = ['red', 'green', 'blue'])
            plt.text(redn[0], red[0], red[0], ha='center', )
            plt.text(redn[1], red[1], red[1], ha='center')
            plt.text(redn[2], red[2], red[2], ha='center')
            plt.text(greenn[0], green[0], green[0], ha='center', )
            plt.text(greenn[1], green[1], green[1], ha='center')
            plt.text(greenn[2], green[2], green[2], ha='center')
            plt.text(nirn[0], nir[0], nir[0], ha='center', )
            plt.text(nirn[1], nir[1], nir[1], ha='center')
            plt.text(nirn[2], nir[2], nir[2], ha='center')
            plt.title(f'RGN Integration Values: Int BW-{samp} samples')
            plt.ylabel('Values')
            plt.show()

        rsum = red[0]+green[0]+nir[0]
        gsum = red[1]+green[1]+nir[1]
        nsum = red[2]+green[2]+nir[2]
        Ra1, Ra2, Ra3 = round((red[0]/rsum),2), round((red[1]/gsum),2), round((red[2]/nsum),2)
        Ga1, Ga2, Ga3 = round((green[0]/rsum),2), round((green[1]/gsum),2), round((green[2]/nsum),2)
        Na1, Na2, Na3 = round((nir[0]/rsum),2), round((nir[1]/gsum),2), round((nir[2]/nsum),2)
        calibration = [[Ra1, Ra2, Ra3], [Ga1, Ga2, Ga3], [Na1, Na2, Na3]]
        print(calibration)
        return calibration

    # Function to integrate the response for each band for calibration
    def integrate_full(self, samp):
        green_total_int = 0
        red_total_int = 0
        nir_total_int = 0
        ga1 = 0 # actually 0
        ga2 = 0
        ga3 = 0 # actually 0
        ra1 = 0
        ra2 = 0 # actually 0
        ra3 = 0
        na1 = 0
        na2 = 0
        na3 = 0
        prev_g = 0
        prev_r = 0
        prev_n = 0
        mg = 0
        mr = 0
        mn = 0
        bg = 0
        br = 0
        bn = 0

        for i, (g, r, n) in enumerate(zip(self.green_values, self.red_values, self.nir_values)):
            # print(f'------{i}--------')
            if i == 0:
                continue
            g = int(g)
            r = int(r)
            n = int(n)

            mg = (g - prev_g) / (i - (i-1))
            mr = (r - prev_r) / (i - (i-1))
            mn = (n - prev_n) / (i - (i-1))
            # print(f'Slope: {mg} : {mr} : {mn}')
            bg = g - (mg * i)
            br = r - (mr * i)
            bn = n - (mn * i)
            # print(f'Y-int: {bg} : {br} : {bn}')
            samples = np.linspace((i-1), i, samp)
            # print(f'Samples: {samples}')

            for s in samples:
                dx = (samples[1]-samples[0])
                green_total_int += dx * (mg*s + bg)
                red_total_int += dx * (mr*s + br)
                nir_total_int += dx * (mn*s + bn)

            prev_g = g
            prev_r = r
            prev_n = n

        # print(f'{green_total_int} : {red_total_int} : {nir_total_int}')

        bands = ['R', 'G', 'N']
        values = [int(red_total_int), int(green_total_int), int(nir_total_int)]

        plt.bar(bands, values, color=['red', 'green', 'blue'])
        plt.text(bands[0], values[0], values[0], ha='center', )
        plt.text(bands[1], values[1], values[1], ha='center')
        plt.text(bands[2], values[2], values[2], ha='center')
        plt.title(f'RGN Integration Values: Int 2-{samp} samples')
        plt.ylabel('Values')
        plt.show()

    # Function to integrate the response for each band for calibration using numpy
    def integrate_np_t3(self, display, stats):

        #Integration variables
        ra1 = 0
        ra2 = 0
        ra3 = 0
        ga1 = 0
        ga2 = 0
        ga3 = 0
        na1 = 0
        na2 = 0
        na3 = 0

        # Integration values from numpy
        #500-600nm
        yr1 = [0, *self.red_values[12:18], 0]
        yg1 = [0, *self.green_values[12:18], 0]
        yn1 = [0, *self.nir_values[12:18], 0]
        ga1 = np.trapz(yr1)
        ga2 = np.trapz(yg1)
        ga3 = np.trapz(yn1)

        # 600-700nm
        yr2 = [0, *self.red_values[23:29], 0]
        yg2 = [0, *self.green_values[23:29], 0]
        yn2 = [0, *self.nir_values[23:29], 0]
        ra1 = np.trapz(yr2)
        ra2 = np.trapz(yg2)
        ra3 = np.trapz(yn2)

        # 700-850nm
        yr3 = [0, *self.red_values[39:48], 0]
        yg3 = [0, *self.green_values[39:48], 0]
        yn3 = [0, *self.nir_values[39:48], 0]
        na1 = np.trapz(yr3)
        na2 = np.trapz(yg3)
        na3 = np.trapz(yn3)

        redn = ['Ra1', 'Ra2', 'Ra3']
        greenn = ['Ga1', 'Ga2', 'Ga3']
        nirn = ['Na1', 'Na2', 'Na3']
        red = [int(ra1), int(ra2), int(ra3)]
        green = [int(ga1), int(ga2), int(ga3)]
        nir = [int(na1), int(na2), int(na3)]

        if display:
            plt.bar(greenn, green, color=['red', 'green', 'blue'])
            plt.bar(redn, red, color=['red', 'green', 'blue'])
            plt.bar(nirn, nir, color=['red', 'green', 'blue'])
            plt.text(redn[0], red[0], red[0], ha='center', )
            plt.text(redn[1], red[1], red[1], ha='center')
            plt.text(redn[2], red[2], red[2], ha='center')
            plt.text(greenn[0], green[0], green[0], ha='center', )
            plt.text(greenn[1], green[1], green[1], ha='center')
            plt.text(greenn[2], green[2], green[2], ha='center')
            plt.text(nirn[0], nir[0], nir[0], ha='center', )
            plt.text(nirn[1], nir[1], nir[1], ha='center')
            plt.text(nirn[2], nir[2], nir[2], ha='center')
            plt.title(f'RGN Integration Values: Numpy')
            plt.ylabel('Values')
            plt.show()

        rsum = red[0] + green[0] + nir[0]
        gsum = red[1] + green[1] + nir[1]
        nsum = red[2] + green[2] + nir[2]

        Ra1, Ra2, Ra3 = round((red[0] / rsum), 2), round((red[1] / gsum), 2), round((red[2] / nsum), 2)
        Ga1, Ga2, Ga3 = round((green[0] / rsum), 2), round((green[1] / gsum), 2), round((green[2] / nsum), 2)
        Na1, Na2, Na3 = round((nir[0] / rsum), 2), round((nir[1] / gsum), 2), round((nir[2] / nsum), 2)
        calibration = [[Ra1, Ra2, Ra3], [Ga1, Ga2, Ga3], [Na1, Na2, Na3]]

        if stats:
            print(f'RED: {self.red_values}')
            print('-' * 30)
            print(f'GREEN: {self.green_values}')
            print('-' * 30)
            print(f'NIR: {self.nir_values}')
            print(f'Y-R: {yr1}')
            print(f'Y-G: {yg1}')
            print(f'Y-N: {yn1}')
            print(ga1, ga2, ga3)
            print(f'Y-R: {yr2}')
            print(f'Y-G: {yg2}')
            print(f'Y-N: {yn2}')
            print(ra1, ra2, ra3)
            print(f'Y-R: {yr3}')
            print(f'Y-G: {yg3}')
            print(f'Y-N: {yn3}')
            print(na1, na2, na3)
            print(calibration)

        return calibration

    # Function to integrate the response for each band for calibration using numpy
    def integrate_np(self, display, stats, prnt):

        #Integration variables
        ra1 = 0
        ra2 = 0
        ra3 = 0
        ga1 = 0
        ga2 = 0
        ga3 = 0
        na1 = 0
        na2 = 0
        na3 = 0

        # Integration values from numpy
        #500-600nm
        yr1 = [0, *self.red_values[6:16], 0]
        yg1 = [0, *self.green_values[6:16], 0]
        yn1 = [0, *self.nir_values[6:16], 0]
        ga1 = np.trapz(yr1)
        ga2 = np.trapz(yg1)
        ga3 = np.trapz(yn1)

        # 600-700nm
        yr2 = [0, *self.red_values[27:38], 0]
        yg2 = [0, *self.green_values[27:38], 0]
        yn2 = [0, *self.nir_values[27:38], 0]
        ra1 = np.trapz(yr2)
        ra2 = np.trapz(yg2)
        ra3 = np.trapz(yn2)

        # 700-850nm
        yr3 = [0, *self.red_values[43:56], 0]
        yg3 = [0, *self.green_values[43:56], 0]
        yn3 = [0, *self.nir_values[43:56], 0]
        na1 = np.trapz(yr3)
        na2 = np.trapz(yg3)
        na3 = np.trapz(yn3)

        redn = ['Ra1', 'Ra2', 'Ra3']
        greenn = ['Ga1', 'Ga2', 'Ga3']
        nirn = ['Na1', 'Na2', 'Na3']
        red = [int(ra1), int(ra2), int(ra3)]
        green = [int(ga1), int(ga2), int(ga3)]
        nir = [int(na1), int(na2), int(na3)]

        if display:
            plt.bar(greenn, green, color=['red', 'green', 'blue'])
            plt.bar(redn, red, color=['red', 'green', 'blue'])
            plt.bar(nirn, nir, color=['red', 'green', 'blue'])
            plt.text(redn[0], red[0], red[0], ha='center', )
            plt.text(redn[1], red[1], red[1], ha='center')
            plt.text(redn[2], red[2], red[2], ha='center')
            plt.text(greenn[0], green[0], green[0], ha='center', )
            plt.text(greenn[1], green[1], green[1], ha='center')
            plt.text(greenn[2], green[2], green[2], ha='center')
            plt.text(nirn[0], nir[0], nir[0], ha='center', )
            plt.text(nirn[1], nir[1], nir[1], ha='center')
            plt.text(nirn[2], nir[2], nir[2], ha='center')
            plt.title(f'RGN Integration Values: JPGS')
            plt.ylabel('Values')
            plt.show()

        rsum = red[0] + green[0] + nir[0]
        gsum = red[1] + green[1] + nir[1]
        nsum = red[2] + green[2] + nir[2]

        Ra1, Ra2, Ra3 = round((red[0] / rsum), 2), round((red[1] / gsum), 2), round((red[2] / nsum), 2)
        Ga1, Ga2, Ga3 = round((green[0] / rsum), 2), round((green[1] / gsum), 2), round((green[2] / nsum), 2)
        Na1, Na2, Na3 = round((nir[0] / rsum), 2), round((nir[1] / gsum), 2), round((nir[2] / nsum), 2)
        calibration = [[Ra1, Ra2, Ra3], [Ga1, Ga2, Ga3], [Na1, Na2, Na3]]

        if stats:
            print(f'RED: {self.red_values}')
            print('-' * 30)
            print(f'GREEN: {self.green_values}')
            print('-' * 30)
            print(f'NIR: {self.nir_values}')
            print(f'Y-R: {yr1}')
            print(f'Y-G: {yg1}')
            print(f'Y-N: {yn1}')
            print(ga1, ga2, ga3)
            print(f'Y-R: {yr2}')
            print(f'Y-G: {yg2}')
            print(f'Y-N: {yn2}')
            print(ra1, ra2, ra3)
            print(f'Y-R: {yr3}')
            print(f'Y-G: {yg3}')
            print(f'Y-N: {yn3}')
            print(na1, na2, na3)
            print(calibration)

        if prnt:
            print(f'        [   R       G       B   ]')
            print(f'RED   = [ {Ra1},    {Ra2},     {Ra3} ]')
            print(f'GREEN = [ {Ga1},    {Ga2},     {Ga3} ]')
            print(f'NIR =   [ {Na1},    {Na2},     {Na3} ]')



        return calibration

# PIXEL CLASS TO STORE LOCATION, VALUES, AND CATEGORY
class pixel_class:

    def __init__(self, location, values):
        self.location = location
        self.values = values
        self.cat_num = 0
        self.subcat_num = 0
        self.subcategory = object

# CATEGORY CLASS TO STORE CAT, SUBCAT, CAT TYPE, SUBCAT TYPE, CAT AV VALUE
class category_class:
    category_list = ['unknown', 'natural', 'manmade', 'noise']
    natural_sub_list = ['unknown', 'vegetation', 'water', 'soil', 'rock']
    manmade_sub_list = ['unknown', 'metal', 'plastic', 'path', 'wood', 'concrete']

    max_subcategories = 0

    def __init__(self):
        self.cat_num = 0
        self.subcat_num = 0
        self.subcat_group = 0
        self.category_type = category_class.category_list[0]
        self.sub_category_type = category_class.natural_sub_list[0]
        self.total_num = 0
        self.average_values = []
        self.pixel_list = []





























