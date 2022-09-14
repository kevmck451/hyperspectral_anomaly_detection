#Hyperspectral is used to create HSI objects and functions
#Kevin McKenzie 2022

from Materials.Materials import Material as m
import matplotlib.gridspec as gridspec
from sklearn import preprocessing
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import numpy as np
import scipy.io
import shutil
import random
import time
import math

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

        self.edit_record = []
        self.file_path = raw_file_path

        try:
            file = raw_file_path.split('.')
            self.file_type = file[1]

        except:
            self.file_type = 'envi'

        if self.file_type == 'mat':
            data = scipy.io.loadmat(self.file_path)
            # print(data)

            name = ''
            for i, k, v in zip(range(len(data.keys())), data.keys(), data.values()):
                # print('k: {}'.format(k))
                # print('v: {}'.format(v))
                if i == 3:
                    name = k

            # print(name)
            data = data.get(name)
            self.data = np.array(data).astype(np.float32) # dtype='u2'

            # print(self.data.shape)
            self.img_x = int(self.data.shape[1])
            self.img_y = int(self.data.shape[0])
            self.img_bands = int(self.data.shape[2])

        if self.file_type == 'envi':
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

    # Function to look if a file exists and writes one if doesnt
    def write_record_file(self, save_to, image_name):
        g = open(save_to + image_name + '-Record.txt', 'w')

        for record in self.edit_record:
            g.writelines(record + '\n')

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

        record = 'Anomaly Synthesis: anomaly added = {}, method = 1, size = {}, location = {}, scale_factor = {}'.format(
            material.material_id, size, location, scale_factor)
        self.edit_record.append(record)

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

        record = 'Anomaly Synthesis: anomaly added = {}, method = 2, size = {}, location = {}'.format(
            material.material_id, size, location)
        self.edit_record.append(record)

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

        record = 'Anomaly Synthesis: anomaly added = {}, method = 3, size = {}, location = {}'.format(
            material.material_id, size, location)
        self.edit_record.append(record)

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

        record = 'Anomaly Synthesis: anomaly added = {}, method = 4, size = {}, location = {}'.format(
            material.material_id, size, location)
        self.edit_record.append(record)

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

        record = 'Anomaly Synthesis: anomaly added = {}, method = 5, size = {}, location = {}, variation = {}'.format(
            material.material_id, size, location, variation)
        self.edit_record.append(record)

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

        record = 'Anomaly Synthesis: anomaly added = None, method = 6, size = {}, location = {}, variation = {}'.format(
            size, location, variation)
        self.edit_record.append(record)

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
        record = 'Anomaly Synthesis: anomaly added = {}, method = 7, size = {}, location = {}, variation = {}, weight = {}'.format(
            material.material_id, size, location, variation, weight)
        self.edit_record.append(record)

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

        record = 'Anomaly Synthesis: anomaly added = {}, method = 8, size = {}, location = {}, variation = {}, weight = {}'.format(
            material.material_id, size, location, variation, weight)
        self.edit_record.append(record)

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

        image = self.file_path.split('/')
        image = image[-1]

        record = 'Image Metadata: image = {}, min = {}, max = {}, average = {}'.format(
            image, min_total, max_total, av_total)
        self.edit_record.append(record)

        return min_max_av_list

    # Function to create a material from single pixel in image
    def graph_spectra_pixel(self, location, title, single):
        values_single = []
        for i in range(self.img_bands):
            values_single.append(self.data[location[1], location[0], i])

        try:
            plt.plot(list(self.wavelengths_dict.values()), values_single, linewidth=2, label = title)
        except:
            x_a = np.linspace(0, self.img_bands, num=self.img_bands)
            plt.plot(x_a, values_single, linewidth=2, label=title)
        plt.xlabel('Bands')
        plt.ylabel('Values')
        plt.title('New Material', fontsize=20)
        # plt.legend(loc='upper right')
        if single:
            plt.show()

    # Function to graph all spectral signature for every pixel in image
    def graph_spectra_all_pixels(self):

        x_list = np.linspace(0, (self.img_x - 1), 100)
        y_list = np.linspace(0, (self.img_y - 1), 100)

        for i in x_list:
            for j in y_list:
                self.graph_spectra_pixel([0, int(i), 0, int(j)], 'Full', False)
        plt.show()

    # Function to create a material from area in image
    def graph_spectra_area_average(self, location):
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

        try:
            plt.plot(list(self.wavelengths_dict.values()), area_values, linewidth=2)
        except:
            x_a = np.linspace(0, self.img_bands, num=self.img_bands)
            plt.plot(x_a, area_values, linewidth=2)
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

        if self.file_type == 'envi':
            im_crop.header_file_dict['samples'] = str(im_crop.img_x)
            im_crop.header_file_dict['lines'] = str(im_crop.img_y)

        image = self.file_path.split('/')
        image = image[-1]

        record = 'Image Edit: edit = crop, image = {}, dimensions = {}'.format(
            image, dimension)
        self.edit_record.append(record)

        return im_crop

    # Function to change the number of bands
    def reduce_bands(self, bands = (300, 1000), index = 66):

        try:
            bottom, top = bands[1], bands[0]
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

            image = self.file_path.split('/')
            image = image[-1]

            record = 'Image Edit: edit = reduce bands, image = {}, bottom = {}, top = {}'.format(
                image, bottom, top)
            self.edit_record.append(record)

            return img

        except:
            img = deepcopy(self)
            new_data = np.zeros(shape=(self.img_y, self.img_x, index))

            for i in range(index):
                new_data[:, :, i] = self.data[:, :, i]

            img.img_bands = index
            img.data = new_data

            return img

    # Function to export image to a file
    def export(self, image_name = 'Untitled'):
        export_im = deepcopy(self)
        save_to = 'Anomaly Files/'
        data = export_im.data.astype(">i2")
        f = open(save_to + image_name, "wb")
        f.write(data)

        self.write_HDR(save_to, image_name)
        self.write_record_file(save_to, image_name)

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

    # Function to display the NDVI
    def display_NDVI(self):
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

        plt.figure(figsize=(10, 15))
        plt.imshow(ndvi_array_temp, cmap=plt.get_cmap("RdYlGn"))
        plt.title('NDVI')
        plt.colorbar()
        plt.axis('off')
        plt.show()

    # Function to display the VARI
    def display_VARI(self):
        # print(self.wavelengths_dict)
        # find bands nearest to NDVI red and nir wavelengths
        # OG bands used 36 (685) / 58 (900)
        # Trying 30-650 / 45-770
        vari_blue = 10
        vari_green = 17
        vari_red = 32
        BLUE = self.data[:, :, vari_blue]
        GREEN = self.data[:, :, vari_green]
        RED = self.data[:, :, vari_red]

        BLUE, GREEN, RED = BLUE.astype('float32'), GREEN.astype('float32'), RED.astype('float32')
        BLUE[BLUE == -50.0], GREEN[GREEN == -50.0], RED[RED == -50.0] = np.nan, np.nan, np.nan

        # calculate ndvi
        vari_array = (GREEN - RED) / (GREEN + RED - BLUE)

        # plot. all of this is matplotlib ---------->
        vari_array_temp = np.zeros(vari_array.shape, dtype=float)
        vari_array_temp[vari_array >= 0.1] = vari_array[vari_array >= 0.1]

        plt.figure(figsize=(18, 8))
        plt.imshow(vari_array, cmap=plt.get_cmap("RdYlGn"))
        plt.title('VARI')
        plt.colorbar()
        plt.axis('off')
        plt.show()

    # Function to display the DVI
    def display_DVI(self):
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
        ndvi_array = NIR - RED

        # plot. all of this is matplotlib ---------->
        ndvi_array_temp = np.zeros(ndvi_array.shape, dtype=float)
        ndvi_array_temp[ndvi_array >= 0.1] = ndvi_array[ndvi_array >= 0.1]

        plt.figure(figsize=(18, 8))
        plt.imshow(ndvi_array_temp, cmap=plt.get_cmap("RdYlGn"))
        plt.title('NDVI')
        plt.colorbar()
        plt.axis('off')
        plt.show()

    # Function to display the NDWI
    def display_NDWI(self):
        # print(self.wavelengths_dict)
        # find bands nearest to NDVI red and nir wavelengths
        # OG bands used 36 (685) / 58 (900)
        # Trying 30-650 / 45-770
        ndwi_green = 17
        ndwi_nir = 52
        GREEN = self.data[:, :, ndwi_green]
        NIR = self.data[:, :, ndwi_nir]
        GREEN, NIR = GREEN.astype('float32'), NIR.astype('float32')
        GREEN[GREEN == -50.0], NIR[NIR == -50.0] = np.nan, np.nan

        # calculate ndwi
        ndwi_array = (GREEN - NIR) / (GREEN + NIR)

        # plot. all of this is matplotlib ---------->
        ndwi_array_temp = np.zeros(ndwi_array.shape, dtype=float)
        ndwi_array_temp[ndwi_array >= 0.1] = ndwi_array[ndwi_array >= 0.1]

        # plt.figure(figsize=(10, 15))
        plt.imshow(ndwi_array_temp, cmap=plt.get_cmap("Blues"))
        plt.title('WDVI')
        plt.colorbar()
        plt.axis('off')
        plt.show()

    # Function to overlay categorized colors onto image
    def display_categories(self):

        # Vegetation & Water ----------------------------
        ndwi_green = 17
        ndvi_red = 32
        ndvi_nir = 52
        GREEN = self.data[:, :, ndwi_green]
        RED = self.data[:, :, ndvi_red]
        NIR = self.data[:, :, ndvi_nir]
        GREEN, RED, NIR =  GREEN.astype('float32'), RED.astype('float32'), NIR.astype('float32')
        GREEN[GREEN == -50.0], RED[RED == -50.0], NIR[NIR == -50.0] = np.nan, np.nan, np.nan

        # calculate ndvi
        ndvi_array = (NIR - RED) / (NIR + RED)
        # calculate ndwi
        ndwi_array = (GREEN - NIR) / (GREEN + NIR)

        # plot. all of this is matplotlib ---------->
        ndwi_array_temp = np.zeros(ndwi_array.shape, dtype=float)
        ndvi_array_temp = np.zeros(ndvi_array.shape, dtype=float)
        threshold_veg = np.max(ndvi_array)
        threshold_wat = np.max(ndwi_array)
        threshold_veg = threshold_veg * 0.35
        threshold_wat = threshold_wat * 0.5
        if type(threshold_veg) != int or type(threshold_veg) != float:
            threshold_veg = 0.35
            threshold_wat = 0.5
        ndvi_array_temp[ndvi_array >= threshold_veg] = ndvi_array[ndvi_array >= threshold_veg]
        ndwi_array_temp[ndwi_array >= threshold_wat] = ndwi_array[ndwi_array >= threshold_wat]
        vegetation = np.zeros(ndvi_array.shape, dtype=float)
        water = np.zeros(ndwi_array.shape, dtype=float)
        vegetation[ndvi_array_temp == 0] = np.nan
        vegetation[ndvi_array_temp > 0] = 1
        water[ndwi_array_temp == 0] = np.nan
        water[ndwi_array_temp > 0] = 1

        # plt.figure(figsize=(10, 10))
        og = plt.imshow(self.data[:, :, 50], cmap = 'gray')
        veg = plt.imshow(vegetation, cmap='Greens')
        wat = plt.imshow(water, cmap='Blues')
        plt.title('Category Overlay')
        plt.colorbar(og, cmap='gray')
        plt.colorbar(veg, cmap='Greens')
        plt.colorbar(wat, cmap='Blues')
        plt.axis('off')
        plt.show()

    # Function to create a material from single pixel in image
    def categorize_pixels(self, band = 'S', variation = 20, iterations = 50):

        if band == 'S': bands = [4, 10, 18, 24, 34, 42, 52, 66] #reduced bands
        elif band == 'M': bands = [2, 6, 10, 14, 18, 22, 34, 42, 52, 56, 66] #reduced bands
        elif band == 'aviris': bands = [14, 23, 32, 47, 69, 98, 131, 184] #aviris images
        elif band == 'mat': bands = [10, 18, 26, 40, 55, 70, 95, 124, 160] #.mat images
        elif band == 'bot': bands = [13, 20, 36, 41, 52, 70, 90, 121] # bot image
        else: bands = [1, 4, 8, 12, 16, 20, 24, 28, 36, 40, 48, 52, 56, 60, 66] #reduced bands

        # PIXEL CLASS TO STORE LOCATION, VALUES, AND CATEGORY
        class pixel_class:
            def __init__(self, location, values):
                self.location = location
                self.values = values
                self.category = 0
                self.subcat = 0

        cll = (variation / 100) - 1
        clh = (variation / 100) + 1
        # x_list = np.linspace(0, (self.img_x - 1), self.img_x, dtype=int)
        # y_list = np.linspace(0, (self.img_y - 1), self.img_y, dtype=int)
        x_list = []
        y_list = []
        # print(self.img_x, self.img_y)
        for i in range(self.img_x): x_list.append(i)
        for i in range(self.img_y): y_list.append(i)
        # if there are bands outside the max num bands, remove
        while np.max(bands) >= self.img_bands:
            bands.pop()

        # INITIATE ALL PIXEL OBJECTS AND SAMPLE VALUES AT POINTS
        self.master_list = []
        pixel_values = []
        for i in x_list:
            for j in y_list:
                for k in bands:
                    pixel_values.append(self.data[j, i, k])

                p = pixel_class([i,j], np.asarray(pixel_values))
                self.master_list.append(p)
                pixel_values = []

        # CATEGORIZE ALL PIXELS BASED ON SIMILARITY

        class category:
            category_list = ['unknown', 'natural', 'manmade', 'noise']
            natural_sub_list = ['unknown', 'vegetation', 'water', 'soil', 'rock']
            manmade_sub_list = ['unknown', 'metal', 'plastic', 'path', 'wood', 'concrete']
            def __init__(self):
                self.cat_num = 0
                self.category = category.category_list[0]
                self.name = 'untitled'
                self.sub_category = category.category_list[0]
                self.total_num = 0


        def is_similar(compare_list, list):
            similar = [1] * len(compare_list)
            comp = []
            for x,y in zip(compare_list, list):
                a, b = (x*cll), (x*clh)
                # print('{} <= {} <= {}'.format(a, y, b))
                if a <= y <= b: comp.append(1)
                else: comp.append(0)
            # print('{} vs {}'.format(comp, similar))
            if comp == similar: return True
            else: return False

        def check_uncat():
            # print('Checking if any Uncategorized')
            for pixel in self.master_list:
                if pixel.category == 0:
                    # print('Found a 0')
                    return True
            # print('Everything is Categorized')
            return False

        # print('Categorizing')
        cat_num = 1
        f = True
        comp = []
        for pixel in self.master_list:
            if f:
                comp = pixel.values
                pixel.category = cat_num
                f = False
                continue
            if is_similar(comp, pixel.values):
                # print('Similar')
                pixel.category = cat_num

        # While Loop -------------------------------------------------------------------------

        while check_uncat():
            print('Cat Num: {}'. format(cat_num))
            cat_num += 1
            comp_w = []
            for pixel in self.master_list:
                if pixel.category == 0:
                    comp_w = pixel.values
                    pixel.category = cat_num
                    break
            for pixel in self.master_list:
                if pixel.category == 0 and is_similar(comp_w, pixel.values):
                    pixel.category = cat_num
            if cat_num > iterations:
                break

        # End of While Loop -------------------------------------------------------------------------

        # Everything is Categorized

        cat_list = []
        for pixel in self.master_list:
            cat_list.append(pixel.category)

        # Create dict of category numbers
        cat_dict = {}
        for i in range(1, cat_num):
            cat_dict.update( { i : cat_list.count(i) } )

        # Sort dict in reverse order based on values
        cat_dict = dict(sorted(cat_dict.items(), key=lambda item: item[1], reverse=True))
        # print(cat_dict)

        new_cat_num_dict = {}
        for i, x in zip(range(1, len(cat_dict)+1), cat_dict.keys()):
            new_cat_num_dict.update( { x : i } )

        # print(new_cat_num_dict)

        for pixel in self.master_list:
            if pixel.category != new_cat_num_dict.get(pixel.category):
                pixel.category = new_cat_num_dict.get(pixel.category)

        cat_list = []
        for pixel in self.master_list:
            cat_list.append(pixel.category)

        # Create dict of category numbers
        cat_dict = {}
        for i in range(1, cat_num):
            cat_dict.update({i: cat_list.count(i)})

        # Sort dict in reverse order based on values
        self.category_data_dict = dict(sorted(cat_dict.items(), key=lambda item: item[1], reverse=True))
        # print(category_data_dict)


        # Create Array with pixels xy with cat valuee
        cat_array = np.zeros((self.data.shape[0], self.data.shape[1], 1), dtype=float)
        for pixel in self.master_list:
            cat_array[pixel.location[1], pixel.location[0], 0] = pixel.category

        # Display Image with Categorized Pixels
        # plt.figure(figsize=(10, 10))
        # og = plt.imshow(self.data[:, :, 50], cmap='gray')

        plt.imshow(cat_array, cmap='nipy_spectral')
        plt.title('Category Overlay')
        plt.colorbar()
        plt.axis('off')
        plt.show()












