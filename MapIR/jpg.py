#Analysis of Data from MapIR Camera
#Kevin McKenzie 2022

import matplotlib.pyplot as plt
from copy import deepcopy
import imageio.v3 as iio
from PIL import Image
import numpy as np
import time
import cv2
import os

save_to_location = '../2 MapIR/Autosaves'

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