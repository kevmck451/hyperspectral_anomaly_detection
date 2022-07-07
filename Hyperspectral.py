#Hyperspectral is used to create HSI objects and functions
#Kevin McKenzie 2022

from Materials.Materials import Material as m
import matplotlib.gridspec as gridspec
from sklearn import preprocessing
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import numpy as np
import shutil
import random

class Hyperspectral:

    wavelength_color_dict = {
        "visible-violet": {'lower': 365, 'upper': 450, 'color': 'violet'},
        "visible-blue": {'lower': 450, 'upper': 485, 'color': 'blue'},
        "visible-cyan": {'lower': 485, 'upper': 500, 'color': 'cyan'},
        "visible-green": {'lower': 500, 'upper': 565, 'color': 'green'},
        "visible-yellow": {'lower': 565, 'upper': 590, 'color': 'yellow'},
        "visible-orange": {'lower': 590, 'upper': 625, 'color': 'orange'},
        "visible-red": {'lower': 625, 'upper': 740, 'color': 'red'},
        "near-infrared": {'lower': 740, 'upper': 1100, 'color': 'gray'},
        "shortwave-infrared": {'lower': 1100, 'upper': 2500, 'color': 'white'}
    }

    def __init__(self, raw_file_path):
        self.anom_record = []

        self.file_path = raw_file_path
        self.data = open(raw_file_path, 'rb') #with open as data
        self.data = np.frombuffer(self.data.read(), ">i2").astype(np.float32) #if with open, remove self from data

        self.open_HDR()

        self.img_x = int(self.header_file_dict.get('samples'))
        self.img_y = int(self.header_file_dict.get('lines'))
        self.img_bands = int(self.header_file_dict.get('bands'))
        self.data = self.data.reshape(self.img_y, self.img_x, self.img_bands)

    # Function to open header files and create variables
    def open_HDR(self):
        self.hdr_file_path = self.file_path + '.hdr'
        hdr_file = open(self.hdr_file_path, 'r')    #errors='ignore'
        self.header_file_dict = {}
        self.wavelengths_dict = {}
        self.wavelengths = []
        self.fwhm_dict = {}

        task_count = 0
        band_num = 1
        fwhm_num = 1

        for i, line in enumerate(hdr_file):
            if task_count == 0:
                self.hdr_title = line.strip().upper()
                task_count = 1

            elif task_count == 1:
                line_split = line.split('=')
                j = line_split[0].strip()
                k = line_split[1].strip()
                self.header_file_dict.update({j: k})
                if j.lower() == 'wavelength':
                    task_count = 2

            elif task_count == 2:
                line = line.split(',')
                wave = line[0].strip()
                if wave.endswith('}'):
                    line = wave.split('}')
                    task_count = 3
                wave = float(line[0])
                self.wavelengths.append(wave)
                wave = round(wave)
                self.wavelengths_dict.update( { band_num : wave } )
                band_num += 1

            elif task_count == 3:
                line_split = line.split('=')
                j = line_split[0].strip()
                k = line_split[1].strip()
                self.header_file_dict.update({j: k})
                task_count = 4

            elif task_count == 4:
                line = line.split(',')
                wave = line[0].strip()
                if wave.endswith('}'):
                    line = wave.split('}')
                    task_count = 5
                wave = float(line[0])
                band_num = 1
                self.fwhm_dict.update({fwhm_num: wave})
                fwhm_num += 1

    # Function to write header files with current info
    def write_HDR(self, save_to, image_name):

        g = open(save_to + image_name + '.hdr', 'w')
        g.writelines('ENVI\n')
        for x, y in self.header_file_dict.items():
            if x == 'fwhm':
                for i, a in enumerate(self.wavelengths):
                    if i == len(self.wavelengths) - 1:
                        d = str(a), ' }\n'
                    else:
                        d = str(a), ' ,\n'
                    g.writelines(d)
                d = x, ' = ', y, '\n'
            else:
                if x == 'samples':
                    y = str(self.img_x)
                if x == 'lines':
                    y = str(self.img_y)
                if x == 'bands':
                    y = str(self.img_bands)
                d = x, ' = ', y, '\n'
            g.writelines(d)

        for i, x in enumerate(self.fwhm_dict.values()):
            if i == len(self.fwhm_dict) - 1:
                d = str(x), ' }\n'
            else:
                d = str(x), ' ,\n'
            g.writelines(d)

    # Function to plot 6 bands of the HSI
    def display_image(self):
        # color_map = 'nipy_spectral'
        # color_map = 'gist_earth'
        # color_map = 'gist_ncar' #fav so far
        color_map = 'Greys_r'
        vmin = -50
        vmax = 7000
        band_list = [50, 39, 24, 18, 12, 8]
        font_size = 12

        plt.figure(figsize=(18, 8))
        plt.subplot(161)
        plt.imshow(self.data[:, :, band_list[0]], cmap=plt.get_cmap(color_map), vmin=vmin, vmax=vmax)
        plt.title('IR-Band: {} nm'.format(self.wavelengths_dict.get(band_list[0])), fontsize=font_size)
        plt.axis('off')
        plt.subplot(162)
        plt.imshow(self.data[:, :, band_list[1]], cmap=plt.get_cmap(color_map), vmin=vmin, vmax=vmax)
        plt.title('R-Band: {} nm'.format(self.wavelengths_dict.get(band_list[1])), fontsize=font_size)
        plt.axis('off')
        plt.subplot(163)
        plt.imshow(self.data[:, :, band_list[2]], cmap=plt.get_cmap(color_map), vmin=vmin, vmax=vmax)
        plt.title('Y-Band: {} nm'.format(self.wavelengths_dict.get(band_list[2])), fontsize=font_size)
        plt.axis('off')
        plt.subplot(164)
        plt.imshow(self.data[:, :, band_list[3]], cmap=plt.get_cmap(color_map), vmin=vmin, vmax=vmax)
        plt.title('G-Band: {} nm'.format(self.wavelengths_dict.get(band_list[3])), fontsize=font_size)
        plt.axis('off')
        plt.subplot(165)
        plt.imshow(self.data[:, :, band_list[4]], cmap=plt.get_cmap(color_map), vmin=vmin, vmax=vmax)
        plt.title('B-Band: {} nm'.format(self.wavelengths_dict.get(band_list[4])), fontsize=font_size)
        plt.axis('off')
        plt.subplot(166)
        plt.imshow(self.data[:, :, band_list[5]], cmap=plt.get_cmap(color_map), vmin=vmin, vmax=vmax)
        plt.title('P-Band: {} nm'.format(self.wavelengths_dict.get(band_list[5])), fontsize=font_size)
        plt.axis('off')
        # cax = plt.axes([0.915, 0.15, 0.02, 0.685]) #left right / up down / width size / length size
        # plt.colorbar(cax=cax)
        plt.show()

    # Function to display the RGB Image
    def display_RGB(self):
        # print(self.wavelengths_dict)
        Ri = 29 #30 #25  # 35 #25 #29 #32
        Gi = 18 #20 #20  # 20 #15 #17 #19
        Bi = 8 #10 #15  # 17 #5 #10 #12

        # get r,g,b arrays
        Ra = self.data[:, :, Ri]
        Ga = self.data[:, :, Gi]
        Ba = self.data[:, :, Bi]

        # set fill values (-9999.) to 0 for each array
        Ra[Ra == -50], Ga[Ga == -50], Ba[Ba == -50] = 0, 0, 0

        # get 8bit arrays for each band
        scale8bit = lambda a: ((a - a.min()) * (1 / (a.max() - a.min()) * 255)).astype('uint8')
        Ra8, Ga8, Ba8 = scale8bit(Ra), scale8bit(Ga), scale8bit(Ba)

        # set rescaled fill pixels back to 0 for each array
        Ra8[Ra == 0], Ga8[Ga == 0], Ba8[Ba == 0] = 0, 0, 0

        # make rgb stack
        rgb_stack = np.zeros((self.img_y, self.img_x, 3), 'uint8')
        rgb_stack[..., 0], rgb_stack[..., 1], rgb_stack[..., 2] = Ra8, Ga8, Ba8

        # apply histogram equalization to each band
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

        # plot. all of this is matplotlib ---------->
        plt.figure(figsize=(18, 8))
        plt.imshow(rgb_stack, cmap=plt.get_cmap(None))
        plt.axis('off')
        plt.show()

    # Function to display the Vegetation Index
    def display_Veg_Index(self):
        # print(self.wavelengths_dict)
        # find bands nearest to NDVI red and nir wavelengths
        # OG bands used 36 (685) / 58 (900)
        # Trying 30-650 / 45-770
        ndvi_red = 32
        ndvi_nir = 52
        RED = self.data[:, :, ndvi_red]
        NIR = self.data[:, :, ndvi_nir]
        RED, NIR = RED.astype('float64'), NIR.astype('float64')
        RED[RED == -50.0], NIR[NIR == -50.0] = np.nan, np.nan

        # calculate ndvi
        ndvi_array = (NIR - RED) / (NIR + RED)

        # plot. all of this is matplotlib ---------->
        ndvi_array_temp = np.zeros(ndvi_array.shape, dtype=float)
        ndvi_array_temp[ndvi_array >= 0.1] = ndvi_array[ndvi_array >= 0.1]

        plt.figure(figsize=(18, 8))
        plt.imshow(ndvi_array_temp, cmap=plt.get_cmap("RdYlGn"))
        plt.title('Vegetation Index')
        plt.colorbar()
        plt.axis('off')
        plt.show()

    # Function to add anomaly using material & image as is
    def add_anomaly_1(self, material, location, size, scale_factor):
        mat = material.map_material_to_image(list(self.wavelengths_dict.values()))
        x_list = [(location[0] - size), (location[0] + size)]
        y_list = [location[1] + size, location[1] - size]
        adjusted = []
        for x in mat:
            adjusted.append(x * scale_factor)

        # Add the anomaly in the right shape
        for i in range(0, len(mat)):
            self.data[y_list[1]:y_list[0], x_list[0]:x_list[1], i] = adjusted[i]
        self.anom_record.append('A1_{}'.format(material.material_id))

    # Function to add anomaly using material spectral mapping to image using image pixel
    def add_anomaly_2(self, material, location, size):
        mat = material.map_material_to_image(list(self.wavelengths_dict.values()))
        x_list = [(location[0] - size), (location[0] + size)]
        y_list = [location[1] + size, location[1] - size]
        min_max = self.pixel_metadata(location[0],location[1])
        adjusted = []
        ad = np.array(mat).reshape(-1, 1)
        scaler = preprocessing.MinMaxScaler(feature_range=(min_max[0], min_max[1]))
        normalizedlist = scaler.fit_transform(ad)
        for i in normalizedlist:
            x = i[0]
            x = round(x)
            adjusted.append(x)

        #Add the anomaly in the right shape
        for i in range(0, len(mat)):
            self.data[y_list[1]:y_list[0], x_list[0]:x_list[1], i] = adjusted[i]
        self.anom_record.append('A2_{}'.format(material.material_id))

    # Function to add anomaly using material spectral mapping to image using image area
    def add_anomaly_3(self, material, location, size):
        mat = material.map_material_to_image(list(self.wavelengths_dict.values()))
        x_list = [(location[0] - size), (location[0] + size)]
        y_list = [location[1] + size, location[1] - size]
        min_max = self.area_metadata(y_list[1], y_list[0], x_list[0], x_list[1])
        adjusted = []
        ad = np.array(mat).reshape(-1, 1)
        scaler = preprocessing.MinMaxScaler(feature_range=(min_max[0], min_max[1]))
        normalizedlist = scaler.fit_transform(ad)
        for i in normalizedlist:
            x = i[0]
            x = round(x)
            adjusted.append(x)

        # Add the anomaly in the right shape
        for i in range(0, len(mat)):
            self.data[y_list[1]:y_list[0], x_list[0]:x_list[1], i] = adjusted[i]
        self.anom_record.append('A3_{}'.format(material.material_id))

    # Function to add anomaly using image's min/max values for material's % reflectivity
    def add_anomaly_4(self, material, location, size):
        mat = material.map_material_to_image(list(self.wavelengths_dict.values()))
        x_list = [(location[0] - size), (location[0] + size)]
        y_list = [location[1] + size, location[1] - size]
        # min_max = self.image_metadata()
        min_max = [-43, 9155, 840.4]
        # min_max = [-50, 32767, 983.8]
        mat_ar = np.array(mat)
        mat_max = mat_ar.max()
        map_max = (mat_max/100) * min_max[1]
        map_max = round(map_max)
        adjusted = []
        ad = np.array(mat).reshape(-1, 1)
        scaler = preprocessing.MinMaxScaler(feature_range=(100, map_max))
        normalizedlist = scaler.fit_transform(ad)
        for i in normalizedlist:
            x = i[0]
            x = round(x)
            adjusted.append(x)

        # Add the anomaly in the right shape
        for i in range(0, len(mat)):
            self.data[y_list[1]:y_list[0], x_list[0]:x_list[1], i] = adjusted[i]
        self.anom_record.append('A4_{}'.format(material.material_id))

    # Function to add anomaly A2 with noise added
    def add_anomaly_5(self, material, location, size, variation):
        mat = material.map_material_to_image(list(self.wavelengths_dict.values()))
        x_list = [(location[0] - size), (location[0] + size)]
        y_list = [location[1] + size, location[1] - size]
        min_max = self.pixel_metadata(location[0], location[1])

        for i in range(y_list[1], y_list[0]):
            for j in range(x_list[0], x_list[1]):
                q = round(variation * min_max[1])
                r = random.randint(-q, q)
                adj_max = min_max[1] + r
                adjusted = []
                ad = np.array(mat).reshape(-1, 1)
                scaler = preprocessing.MinMaxScaler(feature_range=(min_max[0], adj_max))
                normalizedlist = scaler.fit_transform(ad)
                for l in normalizedlist:
                    x = l[0]
                    x = round(x)
                    adjusted.append(x)

                for k in range(0, len(mat)):
                    self.data[i, j, k] = adjusted[k]

                adjusted = []

        # self.anom_record.append('A5_{}'.format(material.material_id))

    # Function to add noise to pixels already in image
    def add_anomaly_6(self, location, size, variation):

        x_list = [(location[0] - size), (location[0] + size)]
        y_list = [location[1] + size, location[1] - size]

        for i in range(y_list[1], y_list[0]):
            for j in range(x_list[0], x_list[1]):
                for k in range(0, self.img_bands):
                    a = self.data[i, j, k]
                    q = abs(round(variation * a))
                    r = random.randint(-q, q)
                    self.data[i, j, k] = a + r

        # self.anom_record.append('A5_{}'.format(material.material_id))

    # Function to add anomaly like A5 but with blurred edges
    def add_anomaly_7(self, material, location, size, variation, weight):
        mat = material.map_material_to_image(list(self.wavelengths_dict.values()))
        x_list = [(location[0] - size), (location[0] + size)]
        y_list = [location[1] + size, location[1] - size]
        min_max = self.pixel_metadata(location[0], location[1])

        for i in range(y_list[1], y_list[0]):
            for j in range(x_list[0], x_list[1]):
                q = round(variation * min_max[1])
                r = random.randint(-q, q)
                adj_max = min_max[1] + r
                adjusted = []
                ad = np.array(mat).reshape(-1, 1)
                scaler = preprocessing.MinMaxScaler(feature_range=(min_max[0], adj_max))
                normalizedlist = scaler.fit_transform(ad)
                for l in normalizedlist:
                    x = l[0]
                    x = round(x)
                    adjusted.append(x)

                for k in range(0, len(mat)):
                    self.data[i, j, k] = adjusted[k]

                adjusted = []

        for i in range(0, len(mat)):
            # Top Edge: E1
            y1 = y_list[1] - 1
            a = self.data[y1, x_list[0]:x_list[1], i] #surround pixels
            b = self.data[y_list[1], x_list[0]:x_list[1], i] #material edge
            w1 = 1
            for x, y in zip(a, b):
                if x > y:
                    w1 = 1 + weight
                else:
                    w1 = 1 - weight
            q = ((w1 * a) + b) / 2
            self.data[y_list[1], x_list[0]:x_list[1], i] = q

            # Bottom Edge: E2
            y2 = y_list[0] - 1
            c = self.data[y2, x_list[0]:x_list[1], i]  # surround pixels
            d = self.data[y_list[0], x_list[0]:x_list[1], i]  # material edge
            w2 = 1
            for x, y in zip(c, d):
                if x > y:
                    w2 = 1 + weight
                else:
                    w2 = 1 - weight
            r = ((w2 * c) + d) / 2
            self.data[y_list[0], x_list[0]:x_list[1], i] = r

            # Left Edge: E3
            x1 = x_list[0] - 1
            y1 = y_list[1] + 1
            y2 = y_list[0] + 1
            e = self.data[ y1 : y2 , x1, i]  # surround pixels
            f = self.data[ y1 : y2, x_list[0], i]  # material edge
            w3 = 1
            for x, y in zip(e, f):
                if x > y:
                    w3 = 1 + weight
                else:
                    w3 = 1 - weight
            s = ((w3 * e) + f) / 2
            self.data[y1 : y2 , x_list[0], i] = s

            # Right Edge: E4
            x2 = x_list[1] - 1
            g = self.data[y1 : y2, x2, i]  # surround pixels
            h = self.data[y1 : y2, x_list[1], i]  # material edge
            w4 = 1
            for x, y in zip(g, h):
                if x > y:
                    w4 = 1 + weight
                else:
                    w4 = 1 - weight
            t = ((w4 * g) + h) / 2
            self.data[y1 : y2, x_list[1], i] = t

        # self.anom_record.append('A5_{}'.format(material.material_id))

    # Function to add anomaly like A5 but to better map the material's profile to images by averaging
    def add_anomaly_8(self, material, location, size, variation, weight):
        mat = material.map_material_to_image(list(self.wavelengths_dict.values()))
        x_list = [(location[0] - size), (location[0] + size)]
        y_list = [location[1] + size, location[1] - size]
        min_max = self.pixel_metadata(location[0], location[1])

        for i in range(y_list[1], y_list[0]):
            for j in range(x_list[0], x_list[1]):
                q = round(variation * min_max[1])
                r = random.randint(-q, q)
                adj_max = min_max[1] + r
                adjusted = []
                ad = np.array(mat).reshape(-1, 1)
                scaler = preprocessing.MinMaxScaler(feature_range=(min_max[0], adj_max))
                normalizedlist = scaler.fit_transform(ad)
                for l in normalizedlist:
                    x = l[0]
                    x = round(x)
                    adjusted.append(x)

                for k in range(0, len(mat)):
                    a = self.data[i, j, k] #OG Value for the pixel about to replace
                    b = adjusted[k]         #adjusted material value
                    wt = weight
                    if weight >= 1:
                        wt = .9999999
                    if a > b:
                        w = 1 + wt
                    elif a <= 0:
                        a = .001
                        w = 1 - wt
                    else:
                        w = 1 - weight
                    q = ((w * a) + b) / 2
                    self.data[i, j, k] = q

                adjusted = []

        # self.anom_record.append('A5_{}'.format(material.material_id))

    # Function to get metadata for single pixel in area we are going to edit
    def pixel_metadata(self, x, y):
        values_single = []
        for i in range(224):
            values_single.append(self.data[y, x, i])

        nums = np.array(values_single)
        min = nums.min()
        max = nums.max()
        av = nums.mean()
        min_max_av_list = [min, max, av]

        return min_max_av_list

    # Function to get metadata for all pixels we are going to edit to see if they are in similar ranges
    def area_metadata(self, y1, y2, x1, x2):
        area_values = []
        for i in range(224):
            for j in range(y1, y2):
                for k in range(x1, x2):
                    area_values.append(self.data[j, k, i])

        nums = np.array(area_values)
        min = nums.min()
        max = nums.max()
        av = nums.mean()
        min_max_av_list = [min, max, av]

        return min_max_av_list

    # Function to get metadata for all pixels in the image
    def image_metadata(self):
        area_values = []
        min = []
        max = []
        av = []

        for i in range(self.img_bands):
            for j in range(self.img_y):
                for k in range(self.img_x):
                    area_values.append(self.data[j, k, i])

            nums = np.array(area_values)
            min.append(nums.min())
            max.append(nums.max())
            av.append(nums.mean())
            area_values = []

        min = np.array(min)
        max = np.array(max)
        av = np.array(av)
        min_total = min.min()
        max_total = max.max()
        av_total = av.mean()
        min_max_av_list = [min_total, max_total, av_total]

        return min_max_av_list

    # Function to create a material from single pixel in image
    def graph_spectra_pixel(self, location, title, single):
        values_single = []
        for i in range(self.img_bands):
            values_single.append(self.data[location[1], location[0], i])

        plt.plot(list(self.wavelengths_dict.values()), values_single, linewidth=2, label = title)
        plt.xlabel('Bands')
        plt.ylabel('Values')
        plt.title('New Material', fontsize=20)
        plt.legend(loc='upper right')
        if single:
            plt.show()

    # Function to create a material from area in image
    def graph_spectra_area(self, location):
        area_values = []
        l = []
        for i in range(self.img_bands):
            for j in range(location[2], location[3]):
                for k in range(location[0], location[1]):
                    l.append(self.data[j, k, i])

            nums = np.array(l)
            av = nums.mean()
            area_values.append(av)
            l = []

        plt.plot(list(self.wavelengths_dict.values()), area_values, linewidth=2)
        plt.xlabel('Bands')
        plt.ylabel('Values')
        plt.title('New Material', fontsize=20)
        # plt.legend(loc='upper right')
        plt.show()

    # Function to create new material text file
    def create_material(self, location, filename, material_info):
        #check to see if file already exists

        values_single = []
        for i in range(224):
            values_single.append(self.data[location[1], location[0], i])
        g = open('Materials/Material Data/' + filename, 'w')
        for i in range(len(m.infonames)):
            d = m.infonames[i], ': ', material_info[i], '\n'
            g.writelines(d)
        g.writelines('\n')
        wave = list(self.wavelengths_dict.values())
        for i in range(len(self.wavelengths)):
            d = str((wave[i]/1000)), ' ', str(values_single[i]), '\n'
            g.writelines(d)
        g.close()

    # Function to crop the image
    def crop(self, dimension):
        im_crop = deepcopy(self)

        new_data = np.zeros(shape = (dimension[3]-dimension[2],dimension[1]-dimension[0], self.img_bands))

        for i in range(self.img_bands):
            new_data[:,:,i] = self.data[dimension[2]:dimension[3], dimension[0]:dimension[1], i]

        im_crop.data = new_data
        im_crop.img_x = dimension[1] - dimension[0]
        im_crop.img_y = dimension[3] - dimension[2]
        im_crop.header_file_dict['samples'] = str(im_crop.img_x)
        im_crop.header_file_dict['lines'] = str(im_crop.img_y)
        return im_crop

    # Function to change the number of bands
    def reduce_bands(self, bottom, top):

        img = deepcopy(self)
        new_wave_dict = {}
        new_wavelengths = []
        new_fwhm_dict = {}

        new_b = 0

        for x in self.wavelengths:
            if top < x < bottom:
                new_wavelengths.append(x)
                new_b += 1

        new_data = np.zeros(shape=(self.img_y, self.img_x, new_b))

        b = 0
        for i, ((x, y), z) in enumerate(zip(self.wavelengths_dict.items(), self.fwhm_dict.values())):
            if top < y < bottom:
                new_wave_dict.update( { (b+1) : y } )
                new_data[:, :, b] = self.data[:, :, x]
                new_fwhm_dict.update({(b + 1): z})
                b += 1

        img.header_file_dict['bands'] = str(new_b)
        img.img_bands = new_b
        img.wavelengths_dict = new_wave_dict
        img.wavelengths = new_wavelengths
        img.fwhm_dict = new_fwhm_dict
        img.data = new_data

        return img

    # Function to export image to a file
    def export(self, image_name):
        export_im = deepcopy(self)
        save_to = 'Anomaly Files/'
        data = export_im.data.astype(">i2")
        f = open(save_to + image_name, "wb")
        f.write(data)

        self.write_HDR(save_to, image_name)




    class Shapes:

        def __init__(self):
            pass








