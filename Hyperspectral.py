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
        self.hdr_file_path = raw_file_path + '.hdr'
        hdr_file = open((raw_file_path + '.hdr'), 'r', errors='ignore')
        self.header_file_dict = {}
        for i in range(1, 20):
            if i == 1:
                line = hdr_file.readline()
                self.hdr_title = line
                self.hdr_title = self.hdr_title.strip().upper()
            if i > 1:
                line = hdr_file.readline()
                line = line.split('=')
                j = line[0]
                j = j.strip()
                k = line[1]
                k = k.strip()
                self.header_file_dict.update({j : k})

        self.img_x = int(self.header_file_dict.get('samples'))
        self.img_y = int(self.header_file_dict.get('lines'))
        self.img_bands = int(self.header_file_dict.get('bands'))

        self.wavelengths_dict = {}
        self.wavelengths = []
        hdr_file.readline()
        for i in range(0, int(self.img_bands)):
            line = hdr_file.readline()
            line = line.split(',')
            wave = line[0].strip()
            if wave.endswith('}'):
                line = wave.split('}')
            wave = float(line[0])
            self.wavelengths.append(wave)
            wave = round(wave)
            self.wavelengths_dict.update({i+1 : wave})
        hdr_file.readline()
        hdr_file.readline()
        #part to read the fwhm

        self.data = self.data.reshape(self.img_y, self.img_x, self.img_bands)

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

    # Function to display the R Band, G Band, B Band, and RGB Images
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

    # Function to display the Vegetation Index: NDVI and SAVI
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

    # Function to add anomaly using material spectral mapping to image using image pixel with noise added
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

    #NOT COMPLETED
    # Function to add anomaly using material spectral mapping to image using image pixel with noise added
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

    # Function to export image to a file
    def export(self, image_name):

        # Export Image to File
        export_im = deepcopy(self)
        save_to = 'Anomaly Files/'
        data = export_im.data.astype(">i2")
        f = open(save_to + image_name, "wb")
        f.write(data)

        #Export HDR File
        # g = open(save_to + image_name + '.hdr', 'w')
        # g.writelines('ENVI\n')
        # for x, y in self.header_file_dict.items():
        #     d = x, ' = ', y, '\n'
        #     g.writelines(d)
        # g.writelines('wavelength = {\n')
        # for x in self.wavelengths:
        #     d = str(x), ' ,\n'
        #     g.writelines(d)

        shutil.copyfile(self.hdr_file_path, (save_to + image_name + '.hdr'))

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

    # Function to create a material from single pixel in image
    def graph_spectra_pixel(self, location, title, single):
        values_single = []
        for i in range(224):
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
        for i in range(224):
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

    def reduce_bands(self):
        #928-966 / 1104-1028 / 1321-1490 / 1768-1974
        #img_bands
        #wavelengths_dict / wavelengths

        img = deepcopy(self)

        print(self.img_bands)
        print(self.wavelengths_dict)
        print(self.wavelengths)

        #create a list of bands to remove for whatever reason
        #A threshold? Pattern?

        #Remove bands function .remove()













    class Shapes:

        def __init__(self):
            pass








