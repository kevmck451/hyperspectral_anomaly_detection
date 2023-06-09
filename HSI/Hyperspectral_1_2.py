#Hyperspectral is used to create HSI objects and functions
#Kevin McKenzie 2022-23

from Materials.Materials import Material as m
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.stats import chi2
from copy import deepcopy
from Hyp_envi import Cube
import imageio.v3 as iio
from PIL import Image
import pandas as pd
import numpy as np
import threading
import datetime
import scipy.io
import spectral
import random
import time
import math
import os

AV_Full = 'Datasets/AVIRIS/f100704t01p00r06rdn_b_sc01_ort_img'
Sat_Mat = ['Datasets/Botswana/Botswana.mat',
         'Datasets/Kennedy Space/KSC.mat',
         'Datasets/Indian Pines/Indian_pines.mat',
         'Datasets/Pavia/Pavia.mat',
         'Datasets/Pavia/PaviaU.mat',
         'Datasets/Salinas/Salinas.mat',]
Sat_Mat_abu = ['Datasets/ABU/abu-airport-1.mat', 'Datasets/ABU/abu-airport-2.mat', 'Datasets/ABU/abu-airport-3.mat',
         'Datasets/ABU/abu-beach-1.mat', 'Datasets/ABU/abu-beach-2.mat', 'Datasets/ABU/abu-beach-3.mat',
         'Datasets/ABU/abu-urban-1.mat', 'Datasets/ABU/abu-urban-2.mat', 'Datasets/ABU/abu-urban-3.mat',
         'Datasets/ABU/abu-urban-4.mat', 'Datasets/ABU/abu-urban-5.mat' ]
Sat_Av_AnomSyn = ['Datasets/Anomaly Files/AV-Crop-3Anom-L', 'Anomaly Files/AV-Crop-3Anom-M', 'Anomaly Files/AV-Crop-7A', 'Anomaly Files/AV-Crop2-8A']
Sat_Aviris = [f'Datasets/AVC Set/AVC-{x}' for x in range(1, 17)]
AC_22 = [f'Datasets/Agricenter/Sum 22/Raw Cubes/ac_22_cube_{x}.bip' for x in range(1, 16)]
AC_22_Pro = [f'Datasets/Agricenter/Sum 22/Processed/ac_22_cube_{x} Processed.bip' for x in range(1, 16)]
AZ_22_Full = 'Datasets/Arizona/AZ Full-LowRes.bip'
AZ_22 = [f'Datasets/Arizona/Cubes/az_cube_{x}.bip' for x in range(1, 15)]
AZ_22_Pro = [f'Datasets/Arizona/Processed/az_cube_{x} Processed.bip' for x in range(1, 15)]

save_to_location = '../1 Anomaly Detection/Autosaves'



# HYPERSPECTRAL CLASS FOR VIEWING, PROCESSING, AND ANALYZING HSI
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

#-------------------------------------------------------------------------------
# LOGISTIC FUNCTIONS
# -------------------------------------------------------------------------------
    def __init__(self, raw_file_path, stats=False):
        self.edit_record = []
        self.subcategorization = False
        self.raw_file_path = raw_file_path
        if stats: print(f'Raw File Path: {self.raw_file_path}')

        #File Type & File Name
        try:
            file = raw_file_path.split('.')
            self.file_type = file[1]
            if stats: print(f'File Type: {self.file_type}')

            try:
                file_name = file[0].split('/')
                self.file_name = file_name[-1]
                file_path = file_name[0:-1]
                self.file_path = file_path[0]
                for i in range(1, len(file_path)):
                    self.file_path = f'{self.file_path}/{file_path[i]}'
                if stats: print(f'File Name: {self.file_name}')
                if stats: print(f'File Path: {self.file_path}')

            except:
                file_name = file[0]
                self.file_name = file_name
                self.file_path = file[0]
                if stats: print(f'File Name: {self.file_name}')
                if stats: print(f'File Path: {self.file_path}')
        except:
            self.file_type = 'envi'
            if stats: print(f'File Type: {self.file_type}')
            try:
                file_name = raw_file_path.split('/')
                self.file_name = file_name[-1]
                if stats: print(f'File Name: {file_name[-1]}')
            except:
                file_name = raw_file_path
                self.file_name = file_name
                if stats: print(f'File Name: {self.file_name}')

        #data, x, y, bands
        if self.file_type == 'bil':
            data = Cube.from_path(self.raw_file_path)
            data = data.read()  # load into memory in native format as numpy array
            # or load as memory map by adding as_mmap=True
            # then only the parts of the cube you access will be read into memory
            # if you need just a small part this can be much faster
            # print(data)
            data = np.array(data).astype(np.float32)

            self.img_x = int(data.shape[2])
            self.img_y = int(data.shape[0])
            self.img_bands = int(data.shape[1])
            if stats: print(f'X: {self.img_x} / Y: {self.img_y} / Bands: {self.img_bands}')

            # reshape_data = np.zeros((self.img_y, self.img_x, self.img_bands), dtype=float)
            self.data = data.transpose((0,2,1))
            self.open_HDR()

        if self.file_type == 'bip':
            data = Cube.from_path(self.raw_file_path)
            data = data.read()  # load into memory in native format as numpy array
            # or load as memory map by adding as_mmap=True
            # then only the parts of the cube you access will be read into memory
            # if you need just a small part this can be much faster
            # print(data)
            self.data = np.array(data).astype(np.float32)

            self.img_x = int(data.shape[1])
            self.img_y = int(data.shape[0])
            self.img_bands = int(data.shape[2])
            if stats: print(f'X: {self.img_x} / Y: {self.img_y} / Bands: {self.img_bands}')

            # reshape_data = np.zeros((self.img_y, self.img_x, self.img_bands), dtype=float)
            # self.data = data.transpose((0,1,2))
            self.open_HDR()


        if self.file_type == 'mat':
            data = scipy.io.loadmat(self.raw_file_path)
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
            if stats: print(f'X: {self.img_x} / Y: {self.img_y} / Bands: {self.img_bands}')

        if self.file_type == 'envi':
            self.data = open(raw_file_path, 'rb') #with open as data
            self.data = np.frombuffer(self.data.read(), ">i2").astype(np.float32) #if with open, remove self from data
            self.open_HDR()

            self.img_x = int(self.header_file_dict.get('samples'))
            self.img_y = int(self.header_file_dict.get('lines'))
            self.img_bands = int(self.header_file_dict.get('bands'))
            self.data = self.data.reshape(self.img_y, self.img_x, self.img_bands)
            if stats: print(f'X: {self.img_x} / Y: {self.img_y} / Bands: {self.img_bands}')

        self.edit_record.append(f'Image Stats : Y Dim-{self.img_y}, X Dim-{self.img_x}, Bands-{self.img_bands}')

    # Function to open header files and create variables
    def open_HDR(self):
        '''
        self.hdr_file_path = self.raw_file_path + '.hdr'
        self.header_file_dict = {}
        self.wavelengths_dict = {}
        self.wavelengths = []
        self.fwhm_dict = {}
        '''


        self.hdr_file_path = self.raw_file_path + '.hdr'
        hdr_file = open(self.hdr_file_path, 'r')    #errors='ignore'
        self.header_file_dict = {}
        self.wavelengths_dict = {}
        self.wavelengths = []
        self.fwhm_dict = {}

        if self.file_type == 'bil' or self.file_type == 'bip':
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
                        k = k.strip('}')
                        k = k.strip('{')
                        wave = k.split(',')
                        for w in wave:
                            val = float(w)
                            self.wavelengths.append(val)
                            val = round(val)
                            self.wavelengths_dict.update({band_num: val})
                            band_num += 1
                        task_count = 3

                elif task_count == 3:
                    line_split = line.split('=')
                    j = line_split[0].strip()
                    k = line_split[1].strip()
                    self.header_file_dict.update({j: k})
                    if j.lower() == 'jwhm':
                        line = line.split(',')
                        wave = line[0].strip()
                        if wave.endswith('}'):
                            line = wave.split('}')
                            task_count = 5
                        wave = float(line[0])
                        band_num = 1
                        self.fwhm_dict.update({fwhm_num: wave})
                        fwhm_num += 1

        else:
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
                    self.wavelengths_dict.update({band_num: wave})

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
    def write_HDR(self, filepath_name):

        g = open(f'{filepath_name}.hdr', 'w')
        g.writelines('ENVI\n')
        for x, y in self.header_file_dict.items():
            if x == 'wavelength':
                # print('WAVE LENGTH')
                # print(self.wavelengths)
                for i, a in enumerate(self.wavelengths):
                    if i == len(self.wavelengths) - 1:
                        d = str(a), ' }\n'
                    else:
                        if i == 0:
                            d = 'wavelength = {', str(a), ' ,'
                        else:
                            d = str(a), ', '
                    g.writelines(d)
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

    # Function to export image to a file
    def export(self, save_name = 'Untitled'):
        import shutil

        filepath = self.file_path.split('/')
        filepath = filepath[:-1]
        # print(filepath)
        for i in range(1, len(filepath)):
            filepath = f'{filepath[0]}/{filepath[i]}'
        filepath = f'{filepath}/{save_name}'

        if not os.path.isdir(filepath):
            os.makedirs(filepath)

        if os.path.isfile(f'{self.file_path}/{self.file_name}.{self.file_type}.png'):
            shutil.copy(f'{self.file_path}/{self.file_name}.{self.file_type}.png', f'{filepath}/{self.file_name} {save_name}.{self.file_type}.png')

        if self.file_type == 'mat': print('File Type is mat: Cannot Export')
        else:
            if self.file_type.lower() == 'envi':
                save_to = f'{filepath}/{self.file_name} {save_name}.{self.file_type}'
                export_im = deepcopy(self)
                data = export_im.data.astype('>i2')
                f = open(save_to, "wb")
                f.write(data)
                self.write_HDR(save_to)
            elif self.file_type.lower() == 'bil':
                save_to = f'{filepath}/{self.file_name} {save_name}.{self.file_type}'
                export_im = deepcopy(self)
                data = export_im.data.astype('>i2')
                f = open(save_to, "wb")
                f.write(data)
                self.write_HDR(save_to)
                # self.write_record_file(save_to, image_name)
            elif self.file_type.lower() == 'bip':
                save_to = f'{filepath}/{self.file_name} {save_name}.{self.file_type}'
                export_im = deepcopy(self)
                data = export_im.data.astype('float32')
                f = open(save_to, "wb")
                f.write(data)
                self.write_HDR(save_to)
                # self.write_record_file(save_to, image_name)
            else: print('Could not Export')

# -------------------------------------------------------------------------------
# EDITING FUNCTIONS
# -------------------------------------------------------------------------------
    # Function to crop the image
    def crop(self, dimension):
        print('Cropping...')
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

        image = self.raw_file_path.split('/')
        image = image[-1]

        record = 'Image Edit: edit = crop, image = {}, dimensions = {}'.format(
            image, dimension)
        self.edit_record.append(record)
        print(f'New Shape: {im_crop.data.shape}')
        return im_crop

    # Function to crop many images from single image
    def crop_many(self):
        name = input('Base Name: ')
        width = int(input('Width: '))
        height = int(input('Height: '))

        while True:
            num = 1
            x1 = int(input('x1 = '))
            x2 = x1 + width
            y1 = int(input('y1 = '))
            y2 = y1 + height

            image = self.crop([x1, x2, y1, y2])
            # image.display_RGB(display=True)
            image.export(f'{name}-{num}')
            num+=1
            exit = input('Crop Another? ')
            if exit == 'n':
                break

    # Function to change the number of bands
    def reduce_bands(self, bands = (300, 1000), index = 66):
        print('Reducing Bands')

        try:
            bottom, top = bands[1], bands[0]
            img = deepcopy(self)
            new_wave_dict = {}
            new_wavelengths = []
            new_fwhm_dict = {}

            new_b = 0

            for x in self.wavelengths:
                if top <= x <= bottom:
                    new_wavelengths.append(x)
                    new_b += 1

            new_data = np.zeros(shape=(self.img_y, self.img_x, new_b))
            self.edit_record.append(f'Reduced Bands : Low Band-{bands[1]}, High Band-{bands[0]}, Bands-{new_b}')

            b = 0
            if len(self.fwhm_dict.values()) > 0:
                for i, ((x, y), z) in enumerate(zip(self.wavelengths_dict.items(), self.fwhm_dict.values())):
                    # print(f'i: {i} / x: {x} / y: {y} / z: {z}')
                    if top <= y <= bottom:
                        new_wave_dict.update( { (b+1) : y } )
                        new_data[:, :, b] = self.data[:, :, x]
                        new_fwhm_dict.update({(b + 1): z})
                        b += 1
            else:
                for i, (x, y) in enumerate(self.wavelengths_dict.items()):
                    # print(f'i: {i} / x: {x} / y: {y}')
                    if top <= y <= bottom:
                        new_wave_dict.update( { (b+1) : y } )
                        new_data[:, :, b] = self.data[:, :, x]
                        b += 1

            img.header_file_dict['bands'] = str(new_b)
            img.img_bands = new_b
            img.wavelengths_dict = new_wave_dict
            img.wavelengths = new_wavelengths
            img.fwhm_dict = new_fwhm_dict
            img.data = new_data

            self.header_file_dict['bands'] = str(new_b)
            self.img_bands = new_b
            self.wavelengths_dict = new_wave_dict
            self.wavelengths = new_wavelengths
            self.fwhm_dict = new_fwhm_dict
            self.data = new_data

            image = self.raw_file_path.split('/')
            image = image[-1]
            print(f'New Shape: {new_data.shape}')

        except:
            # print('First Except')
            img = deepcopy(self)
            new_data = np.zeros(shape=(self.img_y, self.img_x, index))

            for i in range(index):
                new_data[:, :, i] = self.data[:, :, i]

            img.img_bands = index
            img.data = new_data

            self.edit_record.append(f'Reduced Bands : Low Band-{bands[1]}, High Band-{bands[0]}, Bands-{index}')

    # Function to add noise to pixels already in image
    def add_noise(self, location, size, variation):

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

# -------------------------------------------------------------------------------
# INFO FUNCTIONS
# -------------------------------------------------------------------------------
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

        image = self.raw_file_path.split('/')
        image = image[-1]

        return min_max_av_list

# -------------------------------------------------------------------------------
# MAPIR FUNCTIONS
# -------------------------------------------------------------------------------
    # Function to graph all spectral signature for every pixel in image
    def graph_mapir_pika(self, display, save):

        x_list = np.linspace(0, (self.img_x - 1))
        y_list = np.linspace(0, (self.img_y - 1))

        for i in x_list:
            for j in y_list:
                values_single = []
                for k in range(self.img_bands):
                    values_single.append(self.data[int(j), int(i), k])

                plt.plot(list(self.wavelengths_dict.values()), values_single, linewidth=1)
                # plt.ylim((0,50))
                plt.xlabel('Bands')
                plt.ylabel('Counts')

        pika_band = 0
        max_val = 0
        for k in range(self.img_bands):
            values_single = []
            for i in x_list:
                for j in y_list:
                    values_single.append(self.data[int(j), int(i), k])
            max_l = np.max(values_single)
            if max_l > max_val:
                pika_band = k
                max_val = max_l

        wl_list = list(self.wavelengths_dict.values())
        pika_band = wl_list[pika_band]

        band = self.file_name
        band = int(band[0:3])
        plt.vlines(x=[band], colors='black', ls='--', lw=1, ymin=0, ymax=100)
        plt.title(f'MC: {self.file_name} / Pika: {pika_band}nm')

        if save:
            saveas = (f'../../Dropbox/2 Work/1 Optics Lab/2 MapIR/Autosaves/{self.file_name}-Graph')
            plt.savefig(saveas)
            plt.close()
        if display:
            plt.show()

    # Function to graph based on max value
    def graph_all_mapir_pika(self, display, save):

        x_list = np.linspace(0, (self.img_x - 1))
        y_list = np.linspace(0, (self.img_y - 1))
        max_v = 0
        y_values = []
        for i in x_list:
            for j in y_list:
                values_single = []
                for k in range(self.img_bands):
                    values_single.append(int(self.data[int(j), int(i), k]))
                m = np.max(values_single)
                if m > max_v:
                    y_values = values_single
                    max_v = m
        plt.plot(list(self.wavelengths_dict.values()), y_values, linewidth=.5)
        # plt.ylim((0,50))
        plt.xlabel('Bands')
        plt.ylabel('Counts')

        pika_band = 0
        max_val = 0
        for k in range(self.img_bands):
            values_single = []
            for i in x_list:
                for j in y_list:
                    values_single.append(self.data[int(j), int(i), k])
            max_l = np.max(values_single)
            if max_l > max_val:
                pika_band = k
                max_val = max_l

        wl_list = list(self.wavelengths_dict.values())
        pika_band = wl_list[pika_band]

        band = self.file_name
        band = int(band[0:3])
        # plt.vlines(x=[band], colors='black', ls='--', lw=1, ymin=0, ymax=100)
        # plt.title(f'MC: {self.file_name} / Pika: {pika_band}nm')

        if save:
            saveas = (f'../../Dropbox/2 Work/1 Optics Lab/2 MapIR/Autosaves/{self.file_name}-Graph')
            plt.savefig(saveas)
            plt.close()
        if display:
            plt.show()

    # Function to return array
    def max_array(self):

        x_list = np.linspace(0, (self.img_x - 1))
        y_list = np.linspace(0, (self.img_y - 1))
        max_v = 0
        y_values = []
        for i in x_list:
            for j in y_list:
                values_single = []
                for k in range(self.img_bands):
                    values_single.append(int(self.data[int(j), int(i), k]))
                m = np.max(values_single)
                if m > max_v:
                    y_values = values_single
                    max_v = m

        return y_values

    # Function to return the difference between mono and pika
    def mono_pika_comp(self, p):

        x_list = np.linspace(0, (self.img_x - 1))
        y_list = np.linspace(0, (self.img_y - 1))

        pika_band = 0
        max_val = 0

        for k in range(self.img_bands):
            values_single = []
            for i in x_list:
                for j in y_list:
                    values_single.append(self.data[int(j), int(i), k])
            max_l = np.max(values_single)
            if max_l > max_val:
                pika_band = k
                max_val = max_l

        wl_list = list(self.wavelengths_dict.values())
        pika_band = wl_list[pika_band]

        mono_band = self.file_name
        mono_band = int(mono_band[0:3])
        diff = pika_band - mono_band
        if p: print(f'Mono Band: {mono_band} / Pika: {pika_band} / Diff = {diff}')

        return diff

    # Function to display synthesized MapIR Image
    def display_Mapir_Single(self, display, save=False):
        # print(self.wavelengths_dict)
        # 525 - 575 or 550
        # 67 - 91 or 79

        # 625 - 675 or 650
        # 115 - 138 or 127

        # 800 - 875 or 850
        # 196 - 242 or 219

        Ri = 127  # 30 #25  # 35 #25 #29 #32
        Gi = 79  # 20 #20  # 20 #15 #17 #19
        Bi = 219  # 10 #15  # 17 #5 #10 #12

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

        self.rgb_graph = rgb_stack

        if save:
            plt.imshow(rgb_stack, cmap=plt.get_cmap(None))
            plt.axis('off')
            saveas = (f'{save_to_location}/{self.file_name}-RGB')
            plt.savefig(saveas)
            plt.close()

        if display == True:
            plt.imshow(rgb_stack, cmap=plt.get_cmap(None))
            plt.axis('off')
            plt.show()

    # Function to display synthesized MapIR Image
    def display_Mapir_Range(self, display, save=False):
        # print(self.wavelengths_dict)
        # 525 - 575 or 550
        # 67 - 91 or 79

        # 625 - 675 or 650
        # 115 - 138 or 127

        # 800 - 875 or 850
        # 196 - 242 or 219

        Ri = 127  # 30 #25  # 35 #25 #29 #32
        Gi = 79  # 20 #20  # 20 #15 #17 #19
        Bi = 219  # 10 #15  # 17 #5 #10 #12

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

        self.rgb_graph = rgb_stack

        if save:
            plt.imshow(rgb_stack, cmap=plt.get_cmap(None))
            plt.axis('off')
            saveas = (f'{save_to_location}/{self.file_name}-RGB')
            plt.savefig(saveas)
            plt.close()

        if display == True:
            plt.imshow(rgb_stack, cmap=plt.get_cmap(None))
            plt.axis('off')
            plt.show()

# -------------------------------------------------------------------------------
# GRAPH FUNCTIONS
# -------------------------------------------------------------------------------
    # Function to create a material from single pixel in image
    def graph_spectra_pixel(self, location, title, single):
        values_single = []
        for i in range(self.img_bands):
            values_single.append(self.data[location[1], location[0], i])

        # print(self.wavelengths_dict)
        try:
            plt.plot(list(self.wavelengths_dict.values()), values_single, linewidth=2, label=title)

        except:
            x_a = np.linspace(0, self.img_bands, num=self.img_bands)
            plt.plot(x_a, values_single, linewidth=2, label=title)

        plt.xlabel('Bands')
        plt.ylabel('Counts')
        # plt.legend(loc='upper right')
        if single:
            plt.show()

    # Function to graph all spectral signature for every pixel in image
    def graph_spectra_all_pixels(self):

        x_list = np.linspace(0, (self.img_x - 1), 100)
        y_list = np.linspace(0, (self.img_y - 1), 100)

        for i in x_list:
            for j in y_list:
                self.graph_spectra_pixel([int(i), int(j)], 'Full', False)
        plt.show()

    # Function to graph all spectral signature for every pixel in image
    def graph_single_subcategory(self, subcategory):

        print('Subcat Num: {} has {} pixels in it'.format(subcategory, self.subcategory_data_dict.get(subcategory)))

        index = int
        for i, subcat in zip(range(len(self.subcat_master_list)), self.subcat_master_list):
            if subcat.subcat_num == subcategory:
                index = i

        for i, pixel in zip(range(len(self.subcat_master_list[index].pixel_list)), self.subcat_master_list[index].pixel_list):
            title = 'Subcateogry {}: {} Pixels'.format(subcategory, self.subcategory_data_dict.get(subcategory))
            if self.subcategory_data_dict.get(subcategory) > 5000:
                if (i % 20) == 0:
                    location = pixel.location
                    values_single = []
                    for i in range(self.img_bands):
                        values_single.append(self.data[location[1], location[0], i])

                    x_a = np.linspace(0, self.img_bands, num=self.img_bands)
                    plt.plot(x_a, values_single, linewidth=2)
                    plt.xlabel('Bands')
                    plt.ylabel('Values')
                    plt.title(title, fontsize=20)
                else: continue
            elif self.subcategory_data_dict.get(subcategory) > 1000:
                if (i%10) == 0:
                    location = pixel.location
                    values_single = []
                    for i in range(self.img_bands):
                        values_single.append(self.data[location[1], location[0], i])

                    x_a = np.linspace(0, self.img_bands, num=self.img_bands)
                    plt.plot(x_a, values_single, linewidth=2)
                    plt.xlabel('Bands')
                    plt.ylabel('Values')
                    plt.title(title, fontsize=20)
                else:
                    continue
            else:
                location = pixel.location
                values_single = []
                for i in range(self.img_bands):
                    values_single.append(self.data[location[1], location[0], i])

                x_a = np.linspace(0, self.img_bands, num=self.img_bands)
                plt.plot(x_a, values_single, linewidth=2)
                plt.xlabel('Bands')
                plt.ylabel('Values')
                plt.title(title, fontsize=20)
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

    # Function to graph the average values from each subcategory
    def graph_average_subcat(self):

        title = 'Subcateogry Averages'
        for subcat in self.subcat_master_list:
            x_a = np.linspace(0, self.img_bands, num=self.img_bands)
            plt.plot(x_a, subcat.average_values, linewidth=2, label=f'Cat {subcat.subcat_num}')
            plt.xlabel('Bands')
            plt.ylabel('Values')
            plt.legend(loc='upper right')
        plt.title(title, fontsize=20)
        plt.show()


# -------------------------------------------------------------------------------
# DISPLAY FUNCTIONS
# -------------------------------------------------------------------------------
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

        # plt.figure(figsize=(18, 8))
        # plt.subplot(161)
        plt.imshow(self.data[:, :, 0], cmap=plt.get_cmap(color_map))
        # plt.title('IR-Band: {} nm'.format(self.wavelengths_dict.get(band_list[0])), fontsize=font_size)
        plt.axis('off')

        plt.show()

    # Function to display the RGB Image
    def display_RGB(self, display, save=False):
        # print(self.wavelengths_dict)
        r, g, b = 650, 550, 450
        switched_dict = {v: k for k, v in self.wavelengths_dict.items()}
        Ri, Gi, Bi = None, None, None

        while Ri is None:
            Ri = switched_dict.get(r)
            r += 1
        while Gi is None:
            Gi = switched_dict.get(g)
            g += 1
        while Bi is None:
            Bi = switched_dict.get(b)
            b += 1

        # print(f'R: {Ri} / G: {Gi} / B: {Bi}')
        # Ri = 29 #30 #25  # 35 #25 #29 #32
        # Gi = 18 #20 #20  # 20 #15 #17 #19
        # Bi = 8 #10 #15  # 17 #5 #10 #12

        # get r,g,b arrays
        Ra = self.data[:, :, Ri]
        Ga = self.data[:, :, Gi]
        Ba = self.data[:, :, Bi]


        # # set fill values (-9999.) to 0 for each array
        # Ra[Ra == -50], Ga[Ga == -50], Ba[Ba == -50] = 0, 0, 0

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

        self.rgb_graph = rgb_stack

        if save:
            plt.imshow(rgb_stack, cmap=plt.get_cmap(None))
            plt.axis('off')
            saveas = (f'{save_to_location}/{self.file_name}-RGB')
            plt.savefig(saveas)
            plt.close()

        if display == True:
            plt.imshow(rgb_stack, cmap=plt.get_cmap(None))
            plt.axis('off')
            plt.show()

    # Function to overlay categorized colors onto image
    def display_indices(self, display):

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

        if display == True:
            # plt.figure(figsize=(10, 10))
            og = plt.imshow(self.data[:, :, 50], cmap = 'gray')
            veg = plt.imshow(vegetation, cmap='Greens', vmin = 0, vmax= 1, alpha=.5)
            wat = plt.imshow(water, cmap='Blues', vmin = 0, vmax=1, alpha=.5)
            plt.colorbar(og, cmap='gray')
            plt.colorbar(veg, cmap='Greens')
            plt.colorbar(wat, cmap='Blues')
            plt.title('Category Overlay')
            plt.axis('off')
            plt.show()

        else:
            self.graph_indicies = [self.data[:, :, 50], vegetation, water]

    # Function to display all subcategories
    def display_subcategories(self, display, save, cutoff_percent=100):
        if self.subcategorization is False:
            self.categorize_pixels_1()

        cutoff = (len(self.subcategory_data_dict) + 1) * (cutoff_percent / 100)
        cutoff = int((len(self.subcategory_data_dict) + 1) - cutoff)
        # print(cutoff)
        self.category_matrix = np.zeros((self.data.shape[0], self.data.shape[1]), dtype=float)
        for pixel in self.pixel_master_list:
            if pixel.subcat_num > cutoff:
                self.category_matrix[pixel.location[1], pixel.location[0]] = pixel.subcat_num
            else:
                self.category_matrix[pixel.location[1], pixel.location[0]] = 0

        # self.categories_matrix = cat_array

        if display:
            plt.imshow(self.category_matrix, cmap='nipy_spectral')
            plt.title(f'Subcategories-Var: {self.subcat_variation}')
            plt.colorbar()
            plt.axis('off')
            plt.show()

        if save:
            plt.imshow(self.category_matrix, cmap='nipy_spectral')
            plt.title(f'Subcategories-Var: {self.subcat_variation}')
            plt.colorbar()
            plt.axis('off')
            saveas = (f'{save_to_location}/{self.file_name}-Categorize')
            plt.savefig(saveas)
            plt.close()

# -------------------------------------------------------------------------------
# VEGETATION INDICES FUNCTIONS
# -------------------------------------------------------------------------------
    # Function to display the NDVI
    def display_NDVI(self, display, save=False, r=650, n=850):
        print('NDVIing')
        switched_dict = {v: k for k, v in self.wavelengths_dict.items()}
        red, nir = None, None

        while red is None:
            red = switched_dict.get(r)
            r += 1
        while nir is None:
            nir = switched_dict.get(n)
            n += 1

        # print(f'red: {red} / r: {r} / nir: {nir} / n: {n}')
        RED = self.data[:, :, red]
        NIR = self.data[:, :, nir]

        RED, NIR = RED.astype('float'), NIR.astype('float')
        top, bottom = NIR - RED, NIR + RED
        top[top == 0], bottom[bottom == 0] = 0, np.nan

        ndvi_array = np.divide(top, bottom)
        ndvi_array[ndvi_array < 0] = 0
        ndvi_array[ndvi_array > 1] = 1

        self.index_ndvi = np.zeros((self.data.shape[0], self.data.shape[1]), dtype=float)
        self.index_ndvi = ndvi_array

        gt = ground_truth_class(self.raw_file_path)
        self.ndvi_ad_score, self.ndvi_ad_eff = gt.stats(self.index_ndvi)

        if save:
            plt.figure(figsize=(20, 25))
            plt.imshow(ndvi_array, cmap=plt.get_cmap("RdYlGn"))
            plt.title('NDVI')
            plt.colorbar()
            plt.axis('off')
            saveas = f'{save_to_location}/{self.file_name}-NDVI'
            plt.savefig(saveas)
            plt.close()

        if display:
            plt.imshow(ndvi_array, cmap=plt.get_cmap("RdYlGn"))
            plt.title('NDVI')
            plt.colorbar()
            plt.axis('off')
            plt.show()

    # Function to tell average values in an area
    def NDVI_area_values(self, middle_pixel, r=650, n=850):
        print('NDVIing')
        switched_dict = {v: k for k, v in self.wavelengths_dict.items()}
        red, nir = None, None

        while red is None:
            red = switched_dict.get(r)
            r += 1
        while nir is None:
            nir = switched_dict.get(n)
            n += 1

        # print(f'red: {red} / r: {r} / nir: {nir} / n: {n}')
        RED = self.data[:, :, red]
        NIR = self.data[:, :, nir]

        RED, NIR = RED.astype('float'), NIR.astype('float')
        top, bottom = NIR - RED, NIR + RED
        top[top == 0], bottom[bottom == 0] = 0, np.nan

        ndvi_array = np.divide(top, bottom)
        ndvi_array[ndvi_array < 0] = 0
        ndvi_array[ndvi_array > 1] = 1

        self.index_ndvi = np.zeros((self.data.shape[0], self.data.shape[1]), dtype=float)
        self.index_ndvi = ndvi_array

        plus_minus = 40
        x1 = (middle_pixel[0]-plus_minus)
        x2 = (middle_pixel[0]+plus_minus)
        y1 = (middle_pixel[1]-plus_minus)
        y2 = (middle_pixel[1]+plus_minus)

        average_value = ndvi_array[y1:y2, x1:x2].mean()
        print(average_value)

# -------------------------------------------------------------------------------
# PREPROCESSING FUNCTIONS
# -------------------------------------------------------------------------------
    # Function smooth out values but cuts off edges
    def smooth_average(self):
        print('Smoothing Spectra')
        wsize = 5
        data = np.zeros((self.data.shape[0], self.data.shape[1], self.img_bands), dtype=float)

        for y in range(self.data.shape[0]):
            for x in range(self.data.shape[1]):
                values = []
                for i in range(self.img_bands):
                    values.append(self.data[y, x, i])
                weights = np.repeat(1.0, wsize) / wsize
                smoothed = np.convolve(values, weights, 'valid')
                start = max(0, wsize // 2 - len(values) // 2)
                end = self.img_bands - (wsize-1)
                # print(f'S: {start} / F: {end} / B: {self.img_bands}')
                for i in range(self.img_bands):
                    if end - start > i:
                        data[y, x, i] = smoothed[start:end][i]
                    else:
                        data[y, x, i] = values[i]
                values.clear()
        self.data = data

        self.edit_record.append('Smoothing Spectra')

    # Function to make sure all values are greater than zero
    def check_values(self):
        # if value is less than 0, make 0
        print('Removing Negative Values')
        data = self.data
        data[data < 0] = 0
        self.data = data

        self.edit_record.append('Negative values to 0')

    # Function to for preprocessing of HSI: Dimensionality Reduction
    def PCA_comp(self, n_components=10):
        from sklearn.decomposition import PCA

        img = self.data

        # Normalize before PCA
        mean = np.mean(img, axis=(0, 1))
        std = np.std(img, axis=(0, 1))
        std[std == 0] = 0.000001
        normalized_img = (img - mean) / std
        img = normalized_img
        img[np.isnan(img)] = 0  # replace NaN values with 0
        img[np.isinf(img)] = 0  # replace infinity values with 0
        img[img > np.finfo(np.float64).max] = 0  # replace values too large for float64 with 0

        # Flatten the spatial dimensions
        img_flat = img.reshape(-1, img.shape[-1])

        # # Determine the optimal number of components to keep
        # pca = PCA()
        # transformed_data = pca.fit_transform(img_flat)
        # explained_variance_ratio = pca.explained_variance_ratio_

        # Perform PCA with the optimal number of components
        pca = PCA(n_components=n_components)
        print(f'PCA Components: {n_components}')
        self.pca_components = n_components

        transformed_img_flat = pca.fit_transform(img_flat)

        # Reshape the transformed image
        transformed_img = transformed_img_flat.reshape(img.shape[0], img.shape[1], n_components)

        return transformed_img

    # Function to for preprocessing of HSI: Dimensionality Reduction with Threshold
    def PCA_threshold(self, threshold):
        from sklearn.decomposition import PCA

        img = self.data

        # Normalize before PCA
        mean = np.mean(img, axis=(0, 1))
        std = np.std(img, axis=(0, 1))
        normalized_img = (img - mean) / std
        img = normalized_img

        # Flatten the spatial dimensions
        img_flat = img.reshape(-1, img.shape[-1])

        # # Determine the optimal number of components to keep
        pca = PCA()
        transformed_data = pca.fit_transform(img_flat)
        explained_variance_ratio = pca.explained_variance_ratio_

        # Find the number of components needed to reach the threshold
        n_components = 0
        total_variance = 0
        for var in explained_variance_ratio:
            total_variance += var
            n_components += 1
            if total_variance >= threshold:
                break

        print(f'PCA Components: {n_components}')
        self.pca_components = n_components
        # Perform PCA with the optimal number of components
        pca = PCA(n_components=n_components)
        transformed_img_flat = pca.fit_transform(img_flat)

        # Reshape the transformed image
        transformed_img = transformed_img_flat.reshape(img.shape[0], img.shape[1], n_components)

        self.data = transformed_img

    # Function to do all the preprocessing together
    def preprocess(self, bands=(410, 900)):
        runtime_preprocess = time_class('Preprocess')
        self.smooth_average()
        self.reduce_bands(bands=bands)  # Pika settings
        self.check_values()
        runtime_preprocess.stats()

# -------------------------------------------------------------------------------
# ANOMALY DETECTOR FUNCTIONS
# -------------------------------------------------------------------------------
    # Function to be able to perform multiple operations on images at with one process
    def overview(self, display, save, sc_var=20):
        runtime_compare = time_class('Overview')

        # self.display_RGB(display=False)
        # self.categorize_pixels_1(band='pika', variation=sc_var)
        # self.display_NDVI(display=False, save=False)

        thread1 = threading.Thread(target=self.display_RGB, args=(False, False))
        thread2 = threading.Thread(target=self.categorize_pixels_1, args=('pika', sc_var))
        thread3 = threading.Thread(target=self.display_NDVI, args=(False, False))

        thread1.start(), thread2.start(), thread3.start()
        thread1.join(), thread2.join(), thread3.join()

        plt.figure(figsize=(15, 8))
        plt.suptitle(f'Overview\n{self.file_name}')

        plt.subplot(2, 3, 1)
        plt.imshow(self.rgb_graph, cmap=plt.get_cmap(None))
        plt.title('RGB')
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.imshow(self.category_matrix, cmap='nipy_spectral')
        plt.title(f'Pixel Groupings - Var: {self.subcat_variation}')
        plt.colorbar()
        plt.axis('off')

        plt.subplot(2, 3, 3)
        plt.imshow(self.index_ndvi, vmax=1, cmap=plt.get_cmap("RdYlGn"))
        plt.title('NDVI')
        plt.colorbar()
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.title('Spectra of All Pixels')
        x_list = np.linspace(0, (self.img_x - 1), 100)
        y_list = np.linspace(0, (self.img_y - 1), 100)

        for i in x_list:
            for j in y_list:
                self.graph_spectra_pixel([int(i), int(j)], 'Full', False)

        plt.tight_layout(pad=1)

        runtime_compare.stats()

        if save:
            saveas = (f'{save_to_location}/Overview-{self.file_name}')
            plt.savefig(saveas)
            plt.close()

        if display:
            plt.show()

    # Function to categorize each pixel based on spectral similarity
    def categorize_pixels_1(self, band='M', variation=32, iterations=19, av=False):
        print('Categorizing')
        stats = False
        self.subcat_variation = variation
        if band == 'S':
            bands = [4, 10, 18, 24, 34, 42, 52, 66]  # reduced bands
        elif band == 'M':
            bands = [2, 6, 10, 14, 18, 22, 34, 42, 52, 56, 66]  # reduced bands
        elif band == 'aviris':
            bands = [14, 23, 32, 47, 69, 98, 131, 184]  # aviris images
        elif band == 'mat':
            bands = [10, 18, 26, 40, 55, 70, 95, 124, 160]  # .mat images
        elif band == 'bot':
            bands = [13, 20, 36, 41, 52, 70, 90, 121]  # bot image
        elif band == 'pika':
            bands = [0,2,4,8,16,32,64,128,256]  # pika image
        else:
            bands = band  # reduced bands

        if stats: print(f'Bands: {bands}')
        # ----------------------

        cll = (variation / 100) - 1
        clh = (variation / 100) + 1
        x_list = [i for i in range(self.img_x)]
        y_list = [i for i in range(self.img_y)]
        # print(self.img_x, self.img_y)

        # if there are bands outside the max num bands, remove
        while np.max(bands) >= self.img_bands: bands.pop()

        # INITIATE ALL PIXEL OBJECTS AND SAMPLE VALUES AT POINTS
        # print('Initializing Pixel Objects')
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
                if (x * cll) <= y <= (x * clh):
                    comp.append(1)
                else:
                    comp.append(0)
            if comp == similar:
                return True
            else:
                return False

        def check_uncat():
            # print('Checking if any Uncategorized')
            for pixel in self.pixel_master_list:
                if pixel.subcat_num == 0:
                    # print('Found a 0')
                    return True
            # print('Everything is Categorized')
            return False

        # Categorizing ------------------------------------------

        self.subcat_master_list = []
        subcat_num = 1
        c = category_class()
        c.subcat_num = subcat_num
        self.subcat_master_list.append(c)
        f = True
        comp = []
        hh, progress = 0, int((len(self.pixel_master_list)) / 4)

        if stats: print('--First Round Categorizing--')
        for z, pixel in enumerate(self.pixel_master_list):
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

            if stats and z % progress == 0:
                hh += 25
                print(f'{hh}% Categorized')

        category_class.max_subcategories = 1
        # print('Subcat Num: {}'.format(subcat_num))

        if stats: print(f'100% Categorized')

        # ii, prog = 0, (s / 4)
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
            # print('Subcat Num: {}'.format(subcat_num))
        # End of While Loop ---------------
        # print(f'Total Categories: {subcat_num}')
        # Everything is Categorized

        self.subcategory_data_dict = None

        def cat_tally_sort():

            subcat_list = []
            for pixel in self.pixel_master_list:
                subcat_list.append(pixel.subcat_num)
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

            subcat_list = []
            for pixel in self.pixel_master_list:
                subcat_list.append(pixel.subcat_num)

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

        if stats: print('Sorting')
        cat_tally_sort()

        cutoff_percent = 100
        cutoff = (len(self.subcategory_data_dict) + 1) * (cutoff_percent / 100)
        cutoff = int((len(self.subcategory_data_dict) + 1) - cutoff)
        # print(cutoff)
        self.category_matrix = np.zeros((self.data.shape[0], self.data.shape[1]), dtype=float)
        for pixel in self.pixel_master_list:
            if pixel.subcat_num > cutoff:
                self.category_matrix[pixel.location[1], pixel.location[0]] = pixel.subcat_num
            else:
                self.category_matrix[pixel.location[1], pixel.location[0]] = 0

        if av:
            print('Averaging Subcategories')
            average = []
            first = True

            for item in self.subcategory_data_dict.items():
                # print(f'Averaging Subcategory {item[0]}')
                for subcat in self.subcat_master_list:
                    if subcat.subcat_num == item[0]:
                        for pix in subcat.pixel_list:
                            values_single = [self.data[pix.location[1], pix.location[0], i] for i in
                                             range(self.img_bands)]
                            if item[1] == 1:
                                average = values_single
                                continue
                            else:
                                if first:
                                    average = values_single
                                new_list = [((a + b) / 2) for a, b in zip(average, values_single)]
                                average = new_list
                        subcat.average_values = average
                        average = []

    # Function to categorize each pixel based on spectral similarity
    def subcategorize_pixels_2(self, band='pika', variation=20, iterations=19, av=False):
        print('Categorizing')
        self.subcat_variation = variation
        if band == 'S':
            bands = [4, 10, 18, 24, 34, 42, 52, 66]  # reduced bands
        elif band == 'M':
            bands = [2, 6, 10, 14, 18, 22, 34, 42, 52, 56, 66]  # reduced bands
        elif band == 'aviris':
            bands = [14, 23, 32, 47, 69, 98, 131, 184]  # aviris images
        elif band == 'mat':
            bands = [10, 18, 26, 40, 55, 70, 95, 124, 160]  # .mat images
        elif band == 'bot':
            bands = [13, 20, 36, 41, 52, 70, 90, 121]  # bot image
        elif band == 'pika':
            bands = [x for x in range(5, 299, 10)]  # pika image
        else:
            bands = [4, 8, 12, 16, 20, 24, 28, 36, 40, 48, 52, 56, 64, 66]  # reduced bands

        # ----------------------

        cll = (variation / 100) - 1
        clh = (variation / 100) + 1
        x_list = [i for i in range(self.img_x)]
        y_list = [i for i in range(self.img_y)]
        # print(self.img_x, self.img_y)

        # if there are bands outside the max num bands, remove
        while np.max(bands) >= self.img_bands: bands.pop()

        # INITIATE ALL PIXEL OBJECTS AND SAMPLE VALUES AT POINTS
        # print('Initializing Pixel Objects')
        # print(f'Estimated Time: {((len(x_list) * len(y_list)) / 400_500) / 60} mins')
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
                if (x * cll) <= y <= (x * clh):
                    comp.append(1)
                else:
                    comp.append(0)
            if comp == similar:
                return True
            else:
                return False

        def check_uncat():
            # print('Checking if any Uncategorized')
            for pixel in self.pixel_master_list:
                if pixel.subcat_num == 0:
                    # print('Found a 0')
                    return True
            # print('Everything is Categorized')
            return False

        # Categorizing ------------------------------------------

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
        # print('Subcat Num: {}'.format(subcat_num))

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
            # print('Subcat Num: {}'.format(subcat_num))
        # End of While Loop ---------------
        # print(f'Total Categories: {subcat_num}')
        # Everything is Categorized

        self.subcategory_data_dict = None

        def cat_tally_sort():

            subcat_list = []
            for pixel in self.pixel_master_list:
                subcat_list.append(pixel.subcat_num)
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

            subcat_list = []
            for pixel in self.pixel_master_list:
                subcat_list.append(pixel.subcat_num)

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

        if av:
            print('Averaging Subcategories')
            average = []
            first = True

            for item in self.subcategory_data_dict.items():
                # print(f'Averaging Subcategory {item[0]}')
                for subcat in self.subcat_master_list:
                    if subcat.subcat_num == item[0]:
                        for pix in subcat.pixel_list:
                            values_single = [self.data[pix.location[1], pix.location[0], i] for i in
                                             range(self.img_bands)]
                            if item[1] == 1:
                                average = values_single
                                continue
                            else:
                                if first:
                                    average = values_single
                                new_list = [((a + b) / 2) for a, b in zip(average, values_single)]
                                average = new_list
                        subcat.average_values = average
                        average = []

    # Function to display all subcategories filtered
    def sc_filtered(self, display, save, sc_var = 20, cutoff_percent=(40,80)):
        runtime_catfil = time_class('Cat Filter')

        # # bands = [4,24,44,64,84,104,124,144,164,184,204]
        # bands = [x for x in range(20, 251, 10)]
        bands = [0, 5, 10, 20, 30, 50, 70, 100, 130, 170, 210, 250]
        bottom = bands[0]
        top = bands[-1]
        interval = bands[1] - bands[0]

        samples = len(bands)
        print(f'Samples: {samples}')

        self.categorize_pixels_1(band=bands, variation=sc_var)

        self.anom_filter = np.zeros((self.data.shape[0], self.data.shape[1]), dtype=float)
        cutoff_bottom = round(((len(self.subcategory_data_dict)) * (cutoff_percent[0] / 100)))
        cutoff_top = round(((len(self.subcategory_data_dict)) * (cutoff_percent[1] / 100)))
        print(f'B: {cutoff_bottom} / T: {cutoff_top}')

        for x in range(cutoff_bottom, cutoff_top):
            self.anom_filter[self.category_matrix == x] = 1

        gt = ground_truth_class(self.raw_file_path)
        self.filter_ad_score, self.filter_ad_eff = gt.stats(self.anom_filter, stats=False)

        record_catfil = record_class()
        record_catfil.record(
            filename=self.file_name,
            detector='Categories Filtered',
            score=self.filter_ad_score,
            eff=self.filter_ad_eff,
            runtime=runtime_catfil.stats(),
            params={'Cat Var': sc_var, 'Cutoff' : cutoff_percent, 'Range' : (bottom, top, interval), 'Samples' : samples},
            imgstats=self.edit_record
        )

        if display:
            self.anom_filter[self.anom_filter == 0] = np.nan
            og = plt.imshow(self.data[:, :, 50], cmap='gray')
            c1 = plt.colorbar(og, cmap='gray')
            c1.remove()
            anom = plt.imshow(self.anom_filter, cmap='Reds', vmax=1)
            # anom = plt.imshow(self.category_matrix, cmap='nipy_spectral')
            c2 = plt.colorbar(anom, cmap='nipy_spectral')  # 'nipy_spectral'
            c2.remove()
            plt.suptitle('Categories Filtered')
            plt.title(f'AD: {self.filter_ad_score}% / Eff: {self.filter_ad_eff}%')
            plt.colorbar()
            plt.axis('off')
            plt.show()

        if save:
            self.anom_filter[self.anom_filter == 0] = np.nan
            og = plt.imshow(self.data[:, :, 50], cmap='gray')
            c1 = plt.colorbar(og, cmap='gray')
            c1.remove()
            anom = plt.imshow(self.anom_filter, cmap='nipy_spectral')
            # anom = plt.imshow(self.category_matrix, cmap='nipy_spectral')
            c2 = plt.colorbar(anom, cmap='nipy_spectral')  # 'nipy_spectral'
            c2.remove()
            plt.suptitle('Categories Filtered')
            plt.title(f'AD: {self.filter_ad_score}% / Eff: {self.filter_ad_eff}%')
            plt.colorbar()
            plt.axis('off')
            saveas = f'{save_to_location}/{self.file_name}-Filter-CV{sc_var}-CO{cutoff_percent}-R{(bottom, top, interval)}-S{samples}.png'
            plt.savefig(saveas)
            plt.close()

    # Function to display anomalies using RX Global Algorithm
    def Anomaly_RXGlobal(self, display, save, percent=0.999):
        runtime_rxg = time_class('RX Global')
        print('Global RXing')
        bg = spectral.calc_stats(self.data)
        y = spectral.rx(self.data, background=bg)

        n_shape = self.data.shape
        newX = self.data / np.linalg.norm(self.data, ord=2, axis=(0, 1))
        newX = np.reshape(newX, (-1, self.data.shape[2]))
        n_img = newX
        n_img = np.reshape(n_img, (n_shape[0], n_shape[1], n_shape[2]))

        n_bands = n_shape[-1]
        P = chi2.ppf(percent, n_bands)
        y = 1 * (y > P)

        result = y.astype('float64')
        self.anom_rxglobal = np.zeros((self.data.shape[0], self.data.shape[1]), dtype=float)
        self.anom_rxglobal = result

        gt = ground_truth_class(self.raw_file_path)
        self.rxg_ad_score, self.rxg_ad_eff = gt.stats(self.anom_rxglobal)

        record_rxglob = record_class()
        record_rxglob.record(
            filename=self.file_name,
            detector='RX Global',
            score=self.rxg_ad_score,
            eff=self.rxg_ad_eff,
            runtime=runtime_rxg.stats(),
            params={'Percent' : percent},
            imgstats=self.edit_record
        )

        if display:
            result[result == 0] = np.nan
            og = plt.imshow(self.data[:, :, 50], cmap='gray')
            anom = plt.imshow(result, cmap='Reds', vmax=1, alpha=.8)
            c1 = plt.colorbar(og, cmap='gray')
            c2 = plt.colorbar(anom, cmap='Reds')
            c1.remove()
            c2.remove()
            plt.suptitle(f'RX-Global-Percent:{percent}')
            plt.title(f'AD: {self.rxg_ad_score}% / Eff: {self.rxg_ad_eff}%')
            plt.axis('off')
            # plt.colorbar()
            plt.show()

        if save:
            result[result == 0] = np.nan
            og = plt.imshow(self.data[:, :, 50], cmap='gray')
            anom = plt.imshow(result, cmap='Reds', vmax=1, alpha=.8)
            c1 = plt.colorbar(og, cmap='gray')
            c2 = plt.colorbar(anom, cmap='Reds')
            c1.remove()
            c2.remove()
            plt.suptitle(f'RX-Global-Percent:{percent}')
            plt.title(f'AD: {self.rxg_ad_score}% / Eff: {self.rxg_ad_eff}%')
            plt.axis('off')
            saveas = f'{save_to_location}/{self.file_name}-RXGlobal-Per{percent}.png'
            plt.savefig(saveas)
            plt.close()

    # Function to display RX anomalies using spectral Segmented RX
    def Anomaly_RXSeg(self, display, save, seg_size=100):
        runtime_rxs = time_class('RX Segmented')
        print('Seg RXing')

        spectral.settings.envi_support_nonlowercase_params = True
        # self.img = spectral.io.envi.open(hdr_file_path)
        self.img = spectral.open_image(self.raw_file_path + '.hdr')
        self.data = self.img.load()
        self.data = np.where(self.data < 0, 0, self.data)

        # Only with Satelite Data
        # Water Vapour Absorption Band, Water Vapour Absorption Band, Not Illuminated
        # if 105 < self.img.nbands < 225:
        #     w_bands = list(range(105, 120)) + list(range(150, 171)) + list(range(215, 224))
        #     self.data = np.delete(self.data, w_bands, axis=2)

        inc = seg_size
        rxvals = None

        for i in range(0, self.data.shape[0], inc):
            rxrow = None

            for j in range(0, self.data.shape[1], inc):
                d = self.data[i:i + inc, j:j + inc, :]
                rxcol = spectral.rx(d)  # spectral.rx(data, window=(5,211))  Local RX
                rxrow = np.append(rxrow, rxcol, axis=1) if rxrow is not None else rxcol
            rxvals = np.vstack((rxvals, rxrow)) if rxvals is not None else rxrow

        rx_percent = 0.9999 #.999
        nbands = self.data.shape[-1]
        P = chi2.ppf(rx_percent, nbands)
        rx_pd = 1 * (rxvals > P)
        rx_pd = rx_pd.astype('float64')
        self.anom_rx_seg = np.zeros((self.data.shape[0], self.data.shape[1]), dtype=float)
        self.anom_rx_seg[rx_pd > 0] = 1

        gt = ground_truth_class(self.raw_file_path)
        self.rxs_ad_score, self.rxs_ad_eff = gt.stats(self.anom_rx_seg)

        record_rxseg = record_class()
        record_rxseg.record(
            filename=self.file_name,
            detector='RX Segmented',
            score=self.rxs_ad_score,
            eff=self.rxs_ad_eff,
            runtime=runtime_rxs.stats(),
            params={'Seg Size': seg_size},
            imgstats=self.edit_record
        )

        if display:
            self.anom_rx_seg[self.anom_rx_seg == 0] = np.nan
            og = plt.imshow(self.data[:, :, 50], cmap='gray')
            anom = plt.imshow(self.anom_rx_seg, cmap='Reds', vmax=1, alpha=.8)
            c1 = plt.colorbar(og, cmap='gray')
            c2 = plt.colorbar(anom, cmap='Reds')
            c1.remove()
            c2.remove()
            plt.suptitle(f'RX-Segmented-SS:{seg_size}')
            plt.title(f'AD: {self.rxs_ad_score}% / Eff: {self.rxs_ad_eff}%')
            plt.axis('off')
            # plt.colorbar()
            plt.show()

        if save:
            self.anom_rx_seg[self.anom_rx_seg == 0] = np.nan
            og = plt.imshow(self.data[:, :, 50], cmap='gray')
            anom = plt.imshow(self.anom_rx_seg, cmap='Reds', vmax=1, alpha=.8)
            c1 = plt.colorbar(og, cmap='gray')
            c2 = plt.colorbar(anom, cmap='Reds')
            c1.remove()
            c2.remove()
            plt.suptitle(f'RX-Segmented-SS:{seg_size}')
            plt.title(f'AD: {self.rxs_ad_score}% / Eff: {self.rxs_ad_eff}%')
            plt.axis('off')
            saveas = f'{save_to_location}/{self.file_name}-RXSeg-SS{seg_size}.png'
            plt.savefig(saveas)
            plt.close()

     # Function to display anomalies using Isolation Forrest Algorithm
    def Anomaly_IF(self, display, save, contam=0.06):
        runtime_isofor = time_class('Isolation Forest')
        print('Isolation Foresting')
        from sklearn.ensemble import IsolationForest

        # Reshape the data array
        data = self.data.reshape(-1, self.data.shape[-1])

        # Create an instance of the RX anomaly detector and fit it to the data
        detector = IsolationForest(contamination=contam) #.075
        #default is 0.1: 10% will be anomalous
        detector.fit(data)

        # Use the detector to predict anomalies in the data
        labels = detector.predict(data)

        # Create a new image containing only the anomalous points
        anomalous_image = data[labels == -1]

        # Create a new array with the same shape as the original data array
        result = np.zeros_like(data)
        self.anom_isofor = np.zeros((self.data.shape[0], self.data.shape[1]), dtype=float)

        # Copy the anomalous points from the anomalous_image array into the result array
        result[labels == -1] = anomalous_image

        # Take the mean of all bands for each pixel
        result = result.mean(axis=-1)

        # Reshape the result array to the original shape of the data
        result = result.reshape(self.data.shape[0], self.data.shape[1])
        result[result > 0] = 1
        self.anom_isofor = result

        gt = ground_truth_class(self.raw_file_path)
        self.if_ad_score, self.if_ad_eff = gt.stats(self.anom_isofor)

        record_isofor = record_class()
        record_isofor.record(
            filename=self.file_name,
            detector='Isolation Forest',
            score=self.if_ad_score,
            eff=self.if_ad_eff,
            runtime=runtime_isofor.stats(),
            params={'Contam': contam},
            imgstats=self.edit_record
        )

        if display:
            result[result == 0] = np.nan
            og = plt.imshow(self.data[:, :, 50], cmap = 'gray')
            anom = plt.imshow(result, cmap='Reds', vmax=1, alpha=.8)
            c1 = plt.colorbar(og, cmap='gray')
            c2 = plt.colorbar(anom, cmap='Reds')
            c1.remove()
            c2.remove()
            plt.suptitle(f'Isolation Forest-Contam:{contam}')
            plt.title(f'AD: {self.if_ad_score}% / Eff: {self.if_ad_eff}%')
            plt.axis('off')
            # plt.colorbar()
            plt.show()

        if save:
            result[result == 0] = np.nan
            og = plt.imshow(self.data[:, :, 50], cmap='gray')
            anom = plt.imshow(result, cmap='Reds', vmax=1, alpha=.8)
            c1 = plt.colorbar(og, cmap='gray')
            c2 = plt.colorbar(anom, cmap='Reds')
            c1.remove()
            c2.remove()
            plt.suptitle(f'Isolation Forest-Contam:{contam}')
            plt.title(f'AD: {self.if_ad_score}% / Eff: {self.if_ad_eff}%')
            plt.axis('off')
            saveas = f'{save_to_location}/{self.file_name}-IsoFor-C{contam}.png'
            plt.savefig(saveas)
            plt.close()

    # Function to display anomalies using Local Outlier Factor - Fit/Predict Method
    def Anomaly_LOF_fp(self, display, overlay, save, ncomps):
        runtime_loffp = time_class('LOF-FP')
        print('Factoring Local Outliers: Fit-Predict')
        print(f'LOF-fp Start Time: {datetime.datetime.now().strftime("%I:%M %p")}')

        from sklearn.neighbors import LocalOutlierFactor

        neighbor = self.img_bands

        data = self.PCA_comp(n_components=ncomps)
        data = data.reshape(-1, data.shape[-1])

        lof = LocalOutlierFactor(n_neighbors=neighbor) # 100
        lof_scores = lof.fit_predict(data) # -1 or 1 with -1 being outliers

        anom_lof_fp = np.zeros((self.data.shape[0], self.data.shape[1]), dtype=float)
        anom_lof_fp = anom_lof_fp.reshape(-1)
        anom_lof_fp[:] = lof_scores
        anom_lof_fp = anom_lof_fp.reshape(self.data.shape[:2])
        anom_lof_fp[anom_lof_fp == 1] = 0
        anom_lof_fp[anom_lof_fp == -1] = 1
        self.anom_lof_fp = anom_lof_fp

        # print(f' Min: {self.anom_lof.min()} / Max: {self.anom_lof.max()}')

        gt = ground_truth_class(self.raw_file_path)
        self.loffp_ad_score, self.loffp_ad_eff = gt.stats(self.anom_lof_fp, stats=False)

        record_loffp = record_class()
        record_loffp.record(
            filename=self.file_name,
            detector='LOF-FP',
            score=self.loffp_ad_score,
            eff=self.loffp_ad_eff,
            runtime=runtime_loffp.stats(),
            params={'Comps': ncomps, 'Neighbors': neighbor},
            imgstats=self.edit_record
        )

        if overlay:
            anom_lof_fp[anom_lof_fp == 0] = np.nan

        og = plt.imshow(self.data[:, :, 50], cmap='gray')  # 50
        c5 = plt.colorbar(og, cmap='gray')
        c5.remove()
        anom_c1 = plt.imshow(anom_lof_fp, vmax=1, cmap='Reds')  # , alpha=.5
        c6 = plt.colorbar(anom_c1, cmap='Reds')
        c6.remove()
        plt.suptitle('Anomaly LOF: Fit-Predict')
        plt.title(f'AD: {self.loffp_ad_score}% / Eff: {self.loffp_ad_eff}%')
        plt.axis('off')

        if display:
            plt.show()

        if save:
            saveas = f'{save_to_location}/LOF-{self.file_name}-{ncomps}-{neighbor}.png'
            plt.savefig(saveas)
            plt.close()

    # Function to display anomalies using Local Outlier Factor - Decision Function
    def Anomaly_LOF_df(self, display, ncomps, neighbor=500):
        runtime_lofdf = time_class('LOF-DF')
        print('Factoring Local Outliers: Decision Function')
        print(f'LOF-df Start Time: {datetime.datetime.now().strftime("%I:%M %p")}')
        from sklearn.neighbors import LocalOutlierFactor

        data = self.PCA_comp(n_components=ncomps) #0.95
        # data = self.data
        data = data.reshape(-1, data.shape[-1])

        lof = LocalOutlierFactor(n_neighbors=neighbor, novelty=True)  # 20
        lof.fit(data)
        lof_scores = lof.decision_function(data)

        anom_lof_df = np.zeros((self.data.shape[0], self.data.shape[1]), dtype=float)
        anom_lof_df = anom_lof_df.reshape(-1)
        anom_lof_df[:] = lof_scores
        anom_lof_df = anom_lof_df.reshape(self.data.shape[:2])
        anom_lof_df[anom_lof_df >= 0] = 0
        anom_lof_df[anom_lof_df < 0] = 1
        self.anom_lof_df = anom_lof_df

        gt = ground_truth_class(self.raw_file_path)
        self.lofdf_ad_score, self.lofdf_ad_eff = gt.stats(self.anom_lof_df)

        record_lofdf = record_class()
        record_lofdf.record(
            filename=self.file_name,
            detector='LOF-DF',
            score=self.lofdf_ad_score,
            eff=self.lofdf_ad_eff,
            runtime=runtime_lofdf.stats(),
            params={'Comps': ncomps, 'Neighbors': neighbor},
            imgstats=self.edit_record
        )

        if display:
            anom_lof_df[anom_lof_df == 0] = np.nan
            og = plt.imshow(self.data[:, :, 50], cmap='gray')  # 50
            c5 = plt.colorbar(og, cmap='gray')
            c5.remove()
            anom_c1 = plt.imshow(anom_lof_df, cmap='Reds')  # , alpha=.5
            c6 = plt.colorbar(anom_c1, cmap='Reds')
            c6.remove()
            plt.suptitle('Anomaly LOF-df: Decision Function')
            plt.title(f'AD: {self.lofdf_ad_score}% / Eff: {self.lofdf_ad_eff}%')
            plt.axis('off')
            # plt.tight_layout(pad=1)
            plt.show()

    # Function to display anomalies using Support Vector Machine
    def Anomaly_SVM(self, display, save, kern, gamma, ncomps, pca, threshold=False):
        runtime_svm = time_class('SVM')
        print('Supporting Vector Machines')
        print(f'SVM Start Time: {datetime.datetime.now().strftime("%I:%M %p")}')

        current_time = datetime.datetime.now()
        projected_finish_time = current_time + datetime.timedelta(minutes=25)
        print(f'SVM Projected Finish: {projected_finish_time.strftime("%I:%M %p")}')

        from sklearn import svm

        if threshold:
            data = self.PCA_threshold(threshold=pca)  # 0.95
        else:
            data = self.PCA_comp(n_components=ncomps)

        data = data.reshape(-1, data.shape[-1])

        if kern:
            type = 'poly'
            ocsvm = svm.OneClassSVM(kernel='poly', gamma=gamma) # 22 mins / gamma='auto / , nu=0.1
        else:
            type = 'sigmoid'
            ocsvm = svm.OneClassSVM(kernel='sigmoid', gamma=gamma) # 18 mins / gamma='auto' / , nu=0.1
        # ocsvm = svm.OneClassSVM(kernel='rbf', gamma=1) # 21 mins / gamma='auto' 1/bands / , nu=0.1

        ocsvm.fit(data)
        anomalies = ocsvm.predict(data)

        anom_svm = np.zeros((self.data.shape[0], self.data.shape[1]), dtype=float)
        anom_svm = anom_svm.reshape(-1)
        anom_svm[:] = anomalies
        anom_svm = anom_svm.reshape(self.data.shape[:2])
        anom_svm[anom_svm == 1] = 0
        anom_svm[anom_svm == -1] = 1
        self.anom_svm = anom_svm

        # print(f' Min: {self.anom_lof.min()} / Max: {self.anom_lof.max()}')

        gt = ground_truth_class(self.raw_file_path)
        self.svm_ad_score, self.svm_ad_eff = gt.stats(self.anom_svm)

        record_svm = record_class()
        record_svm.record(
            filename=self.file_name,
            detector='SVM',
            score=self.svm_ad_score,
            eff=self.svm_ad_eff,
            runtime=runtime_svm.stats(),
            params={'Comps': ncomps, 'Type' : type, 'Kern': kern, 'Gamma' : gamma},
            imgstats=self.edit_record
        )

        if display:
            anom_svm[anom_svm == 0] = np.nan
            og = plt.imshow(self.data[:, :, 50], cmap='gray')  # 50
            c5 = plt.colorbar(og, cmap='gray')
            c5.remove()
            anom_c1 = plt.imshow(anom_svm, cmap='Reds')  # , alpha=.5
            c6 = plt.colorbar(anom_c1, cmap='Reds')
            c6.remove()
            plt.suptitle('Anomaly SVM')
            plt.title(f'AD: {self.svm_ad_score}% / Eff: {self.svm_ad_eff}%')
            plt.axis('off')
            # plt.tight_layout(pad=1)
            plt.show()

        if save:
            # anom_svm[anom_svm == 1] = np.nan
            # og = plt.imshow(self.data[:, :, 50], cmap='gray')  # 50
            # c5 = plt.colorbar(og, cmap='gray')
            # c5.remove()
            anom_c1 = plt.imshow(anom_svm, cmap='Reds')  # , alpha=.5
            c6 = plt.colorbar(anom_c1, cmap='Reds')
            c6.remove()
            plt.suptitle('Anomaly SVM')
            plt.title(f'AD: {self.svm_ad_score}% / Eff: {self.svm_ad_eff}%')
            plt.axis('off')
            saveas = f'{save_to_location}/{self.file_name}-SVM-{type}-{ncomps}-{gamma}.png'
            plt.savefig(saveas)
            plt.close()

    # Function to for preprocessing of HSI: Dimensionality Reduction
    def Anomaly_PCA(self, display, save, stats):
        from sklearn.decomposition import PCA

        img = self.data

        # Flatten the spatial dimensions
        img_flat = img.reshape(-1, img.shape[-1])
        n_components = 2

        # Perform PCA to reduce the dimensionality of the data
        pca = PCA(n_components=n_components)
        pca.fit(img_flat)
        data_pca = pca.transform(img_flat)

        # Calculate the reconstruction error for each point
        reconstructed_img = pca.inverse_transform(data_pca)
        reconstructed_img = reconstructed_img.reshape(img.shape)
        reconstruction_error = np.mean((img - reconstructed_img) ** 2, axis=1)

        # Plot the original data and the PCA projection
        if stats:
            box = 100
            plt.figure(figsize=(24 , 10))
            plt.suptitle(f'{self.file_name}-PCA-{n_components}.png')
            plt.subplot(1,3,1)
            plt.scatter(img[:, 0], img[:, 1], color='blue', s=.1) # OG Data
            # plt.xlim((-1*box, box))
            # plt.ylim((-1*box, box))
            plt.title(f'Original Data')
            plt.subplot(1, 3, 2)
            plt.scatter(data_pca[:, 0], data_pca[:, 1], color='red', s=.1) # DR Data
            # plt.xlim((-1*box, box))
            # plt.ylim((-1*box, box))
            plt.title(f'PCA Projection')
            plt.subplot(1, 3, 3)
            plt.hist(reconstruction_error, bins=20) # Reconstruction error
            plt.title(f'Reconstruction Error')
            plt.tight_layout(pad=1)
            saveas = f'{save_to_location}/{self.file_name}-PCA-{n_components}.png'
            # plt.savefig(saveas)
            # plt.close()
            plt.show()

        # Identify anomalies as points with high reconstruction error
        # threshold = 1 * np.std(reconstruction_error) #lowering the value makes more sensative
        threshold = .001

        anomalies = np.where(reconstruction_error > threshold) #tuple that contains index of anomalies

        anom_pca = np.zeros((self.data.shape[0], self.data.shape[1]), dtype=float)
        anom_pca[anomalies[0], anomalies[1]] = 1

        self.anom_pca = anom_pca

        gt = ground_truth_class(self.raw_file_path)
        self.pca_ad_score, self.pca_ad_eff = gt.stats(self.anom_pca)

        if display:
            anom_pca[anom_pca == 0] = np.nan
            og = plt.imshow(self.data[:, :, 50], cmap='gray')  # 50
            c5 = plt.colorbar(og, cmap='gray')
            c5.remove()
            anom_c1 = plt.imshow(anom_pca, cmap='Reds')  # , alpha=.5
            c6 = plt.colorbar(anom_c1, cmap='Reds')
            c6.remove()
            plt.suptitle('Anomaly PCA')
            plt.title(f'AD: {self.pca_ad_score}% / Eff: {self.pca_ad_eff}%')
            plt.axis('off')
            # plt.tight_layout(pad=1)
            plt.show()

        if save:
            # anom_svm[anom_svm == 1] = np.nan
            # og = plt.imshow(self.data[:, :, 50], cmap='gray')  # 50
            # c5 = plt.colorbar(og, cmap='gray')
            # c5.remove()
            anom_c1 = plt.imshow(anom_pca, cmap='Reds')  # , alpha=.5
            c6 = plt.colorbar(anom_c1, cmap='Reds')
            c6.remove()
            plt.suptitle('Anomaly PCA')
            plt.title(f'AD: {self.pca_ad_score}% / Eff: {self.pca_ad_eff}%')
            plt.axis('off')
            saveas = f'{save_to_location}/{self.file_name}-PCA-{n_components}.png'
            plt.savefig(saveas)
            plt.close()

    # Function is trained using binary cross-entropy loss to minimize the difference between the reconstructed output and the original input.
    def Anomaly_Autoencoder(self, dp, sv, ep=10, bs=32, tv=95, loss='binary_crossentropy', optimizer='adam'):
        runtime_autoencoder = time_class('Autoencoder')
        import tensorflow as tf

        data = self.data
        data = data.reshape(-1, data.shape[-1])

        # Define the autoencoder model using TensorFlow
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(data.shape[1],)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(data.shape[1], activation='sigmoid')
        ])

        # Compile the model with a reconstruction loss
        model.compile(optimizer=optimizer, loss=loss)

        # Train the model on the hyperspectral image data
        model.fit(data, data, epochs=ep, batch_size=bs, verbose=0)

        # Use the trained model to predict the reconstructed outputs for the hyperspectral image data
        reconstructed_outputs = model.predict(data)

        # Calculate the reconstruction error for each data point
        reconstruction_error = np.mean(np.square(reconstructed_outputs - data), axis=1)
        threshold = np.percentile(reconstruction_error, tv)

        # Identify the anomalous data points by comparing the reconstruction error to the threshold
        anomalies = np.where(reconstruction_error > threshold)[0]
        anom_autoencode = np.zeros((self.data.shape[0], self.data.shape[1]), dtype=float)
        anom_autoencode[np.unravel_index(anomalies, anom_autoencode.shape)] = 1
        self.anom_autoencode = anom_autoencode

        gt = ground_truth_class(self.raw_file_path)
        self.autoencode_ad_score, self.autoencode_ad_eff = gt.stats(self.anom_autoencode)

        record_autoencoder = record_class()
        record_autoencoder.record(
            filename=self.file_name,
            detector='Autoencoder',
            score=self.svm_ad_score,
            eff=self.svm_ad_eff,
            runtime=runtime_autoencoder.stats(),
            params={'Optimizer': optimizer, 'Loss': loss, 'Ep': ep, 'Bs': bs, 'TV':tv},
            imgstats=self.edit_record
        )

        if dp:
            anom_autoencode[anom_autoencode == 0] = np.nan
            og = plt.imshow(self.data[:, :, 50], cmap='gray')  # 50
            c5 = plt.colorbar(og, cmap='gray')
            c5.remove()
            anom_c1 = plt.imshow(anom_autoencode, cmap='Reds')  # , alpha=.5
            c6 = plt.colorbar(anom_c1, cmap='Reds')
            c6.remove()
            plt.suptitle(f'Anomaly AutoEncoder\nO{optimizer}-L{loss}-ep{ep}-bs{bs}-tv{tv}')
            plt.title(f'AD: {self.autoencode_ad_score}% / Eff: {self.autoencode_ad_eff}%')
            plt.axis('off')
            # plt.tight_layout(pad=1)
            plt.show()

        if sv:
            anom_autoencode[anom_autoencode == 0] = np.nan
            og = plt.imshow(self.data[:, :, 50], cmap='gray')  # 50
            c5 = plt.colorbar(og, cmap='gray')
            c5.remove()
            anom_c1 = plt.imshow(anom_autoencode, cmap='Reds')  # , alpha=.5
            c6 = plt.colorbar(anom_c1, cmap='Reds')
            c6.remove()
            plt.suptitle(f'Anomaly AutoEncoder\nO{optimizer}-L{loss}-ep{ep}-bs{bs}-tv{tv}')
            plt.title(f'AD: {self.autoencode_ad_score}% / Eff: {self.autoencode_ad_eff}%')
            plt.axis('off')
            # plt.tight_layout(pad=1)
            saveas = f'{save_to_location}/{self.file_name}-AE-ep{ep}-bs{bs}-tv{tv}.png'
            plt.savefig(saveas)
            plt.close()

# -------------------------------------------------------------------------------
# STACK FUNCTIONS
# -------------------------------------------------------------------------------
    # Function to combine AD's: Filter, IF, RXSeg
    def Anomaly_Stack_1(self, display=True, overlay=False, save=False):
        runtime_s1 = time_class('Stack 1')
        start_time = time.time()
        sc_var = 20
        cutoff_percent = (40, 80)
        contamination = 0.09
        seg_size = self.img_bands
        # bands = [x for x in range(20, 251, 10)]
        # bands = [4, 6, 10, 19, 28, 36, 46, 56, 76, 86, 96, 108, 120, 145, 180, 205, 235, 270]
        # bands = [0, 5, 10, 15, 25, 35, 45, 65, 85, 105, 135, 165, 195, 230, 270]
        bands = [0, 5, 10, 20, 30, 50, 70, 100, 130, 170, 210, 250]
        # bands = [5, 10, 20, 30, 50, 80, 130, 190, 250]

        bottom = bands[0]
        top = bands[-1]
        interval = '[0, 5, 10, 20, 30, 50, 70, 100, 130, 170, 210, 250]'
        samples = len(bands)

        thread1 = threading.Thread(target=self.sc_filtered, args=(False, False, bands, sc_var, cutoff_percent))
        thread2 = threading.Thread(target=self.Anomaly_IF, args=(False, False, contamination))
        thread3 = threading.Thread(target=self.Anomaly_RXSeg, args=(False, False, seg_size))

        thread1.start(), thread2.start(), thread3.start()
        thread1.join(), thread2.join(), thread3.join()

        anom_subcat = self.anom_filter
        anom_isofor = self.anom_isofor
        anom_rx_seg = self.anom_rx_seg

        anom_collection = np.zeros((self.data.shape[0], self.data.shape[1]), dtype=float)
        self.anom_stack_1 = np.zeros((self.data.shape[0], self.data.shape[1]), dtype=float)
        mask_one = mask_two = mask_three = np.zeros((self.data.shape[0], self.data.shape[1]), dtype=float)

        mask_one = np.logical_and(anom_subcat, anom_isofor)
        mask_two = np.logical_and(anom_subcat, anom_rx_seg)
        mask_three = np.logical_and(anom_rx_seg, anom_isofor)

        anom_collection[mask_one] = 1
        anom_collection[mask_two] = 1
        anom_collection[mask_three] = 1

        self.anom_stack_1[anom_collection == True] = 1
        self.anom_stack_1[anom_collection == False] = 0

        gt = ground_truth_class(self.raw_file_path)
        self.as1_ad_score, self.as1_ad_eff = gt.stats(self.anom_stack_1)

        record_stack1 = record_class()
        record_stack1.record(
            filename=self.file_name,
            detector='Stack 1',
            score=self.as1_ad_score,
            eff=self.as1_ad_eff,
            runtime=runtime_s1.stats(),
            params={'Cat Var': sc_var,
                    'Cutoff' : cutoff_percent,
                    'Range' : (bottom, top, interval),
                    'Samples' : samples,
                    'Contam' : contamination,
                    'Seg Size' : seg_size},
            imgstats=self.edit_record
        )

        cmap1 = cmap2 = cmap3 = cmap4 = cmap5 = 'Blues'
        if overlay:
            anom_subcat[anom_subcat == 0] = np.nan
            anom_isofor[anom_isofor == 0] = np.nan
            anom_rx_seg[anom_rx_seg == 0] = np.nan
            anom_collection[anom_collection == False] = np.nan
            cmap1 = cmap2 = cmap3 = cmap4 = 'brg'

        total_time = round(((time.time() - start_time) / 60), 1)
        # print(f'Anomaly Stack Time: {total_time} mins')

        plt.figure(figsize=(15, 8))
        plt.suptitle(f'Anomaly Stack 1 Comparison - Time: {total_time} mins\n{self.file_name}')
        plt.subplot(1, 4, 1)
        og = plt.imshow(self.data[:, :, 50], cmap='gray')
        c1 = plt.colorbar(og, cmap='gray')
        c1.remove()
        anom_sc = plt.imshow(anom_subcat, cmap=cmap1, vmax=1) #, alpha=.5
        c2 = plt.colorbar(anom_sc, cmap=cmap1)
        c2.remove()
        plt.title(f'Filtered - Var: {sc_var}/CO: {cutoff_percent}\nADS: {self.filter_ad_score}% / ADE: {self.filter_ad_eff}%')
        plt.axis('off')

        plt.subplot(1, 4, 2)
        og = plt.imshow(self.data[:, :, 50], cmap='gray')
        c3 = plt.colorbar(og, cmap='gray')
        c3.remove()
        anom_if = plt.imshow(anom_isofor, cmap=cmap2, vmax=1) #, alpha=.5
        c4 = plt.colorbar(anom_if, cmap=cmap2)
        c4.remove()
        plt.title(f'Iso For - Contam: {contamination}\nADS: {self.if_ad_score}% / ADE: {self.if_ad_eff}%')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        og = plt.imshow(self.data[:, :, 50], cmap='gray')
        c3 = plt.colorbar(og, cmap='gray')
        c3.remove()
        anom_rxs = plt.imshow(anom_rx_seg, cmap=cmap3, vmax=1)  # , alpha=.5
        c4 = plt.colorbar(anom_rxs, cmap=cmap3)
        c4.remove()
        plt.title(f'RX-Seg - 1% Anom\nADS: {self.rxs_ad_score}% / ADE: {self.rxs_ad_eff}%')
        plt.axis('off')

        plt.subplot(1, 4, 4)
        og = plt.imshow(self.data[:, :, 50], cmap='gray')
        c5 = plt.colorbar(og, cmap='gray')
        c5.remove()
        anom_c1 = plt.imshow(anom_collection, cmap=cmap4) #, alpha=.5
        c6 = plt.colorbar(anom_c1, cmap=cmap4)
        c6.remove()
        plt.title(f'AD Stack 1\nADS: {self.as1_ad_score}% / ADE: {self.as1_ad_eff}%')
        plt.axis('off')
        plt.tight_layout(pad=1)

        if display:
            plt.show()

        if save:
            saveas = saveas = f'{save_to_location}/{self.file_name}-Stack1.png'
            plt.savefig(saveas)
            plt.close()

    # Function to combine AD's: Filter, IF, LOF-fp
    def Anomaly_Stack_1b(self, display=True, overlay=False, save=False):
        runtime_s1b = time_class('Stack 1b')
        start_time = time.time()
        sc_var = 32
        cutoff_percent = (30, 100)
        contamination = 0.09
        ncomps = 3
        neighbor = 1000

        thread1 = threading.Thread(target=self.sc_filtered, args=(False, 'pika', sc_var, cutoff_percent))
        thread2 = threading.Thread(target=self.Anomaly_IF, args=(False, contamination))
        thread3 = threading.Thread(target=self.Anomaly_LOF_fp, args=(False, False, ncomps, neighbor))

        thread1.start(), thread2.start(), thread3.start()
        thread1.join(), thread2.join(), thread3.join()

        anom_subcat = self.anom_filter
        anom_isofor = self.anom_isofor
        anom_lof_fp = self.anom_lof_fp

        anom_collection = np.zeros((self.data.shape[0], self.data.shape[1]), dtype=float)
        self.anom_stack_1b = np.zeros((self.data.shape[0], self.data.shape[1]), dtype=float)
        mask_one = mask_two = mask_three = np.zeros((self.data.shape[0], self.data.shape[1]), dtype=float)

        mask_one = np.logical_and(anom_subcat, anom_isofor)
        mask_two = np.logical_and(anom_subcat, anom_lof_fp)
        mask_three = np.logical_and(anom_lof_fp, anom_isofor)

        anom_collection[mask_one] = 1
        anom_collection[mask_two] = 1
        anom_collection[mask_three] = 1

        self.anom_stack_1b[anom_collection == True] = 1
        self.anom_stack_1b[anom_collection == False] = 0

        gt = ground_truth_class(self.raw_file_path)
        self.as1b_ad_score, self.as1b_ad_eff = gt.stats(self.anom_stack_1b)

        record_stack1b = record_class()
        record_stack1b.record(
            filename=self.file_name,
            detector='Stack 1b',
            score=self.as1b_ad_score,
            eff=self.as1b_ad_eff,
            runtime=runtime_s1b.stats(),
            params={'Cat Var': sc_var, 'Cutoff': cutoff_percent, 'Contam': contamination, 'Comps': ncomps, 'Neighbor':neighbor},
            imgstats=self.edit_record
        )

        cmap1 = cmap2 = cmap3 = cmap4 = 'Blues'

        if overlay:
            anom_subcat[anom_subcat == 0] = np.nan
            anom_isofor[anom_isofor == 0] = np.nan
            anom_lof_fp[anom_lof_fp == 0] = np.nan
            anom_collection[anom_collection == False] = np.nan
            cmap1 = cmap2 = cmap3 = cmap4 = 'brg'

        total_time = round(((time.time() - start_time) / 60), 1)
        print(f'Anomaly Stack Time: {total_time} mins')

        plt.figure(figsize=(15, 8))
        plt.suptitle(f'Anomaly Stack 2 Comparison - Time: {total_time} mins\n{self.file_name}')
        plt.subplot(1, 4, 1)
        og = plt.imshow(self.data[:, :, 50], cmap='gray')
        c1 = plt.colorbar(og, cmap='gray')
        c1.remove()
        anom_sc = plt.imshow(anom_subcat, cmap=cmap1, vmax=1) #, alpha=.5
        c2 = plt.colorbar(anom_sc, cmap=cmap1)
        c2.remove()
        plt.title(f'Filtered - Var: {sc_var}/CO: {cutoff_percent}\nADS: {self.filter_ad_score}% / ADE: {self.filter_ad_eff}%')
        plt.axis('off')

        plt.subplot(1, 4, 2)
        og = plt.imshow(self.data[:, :, 50], cmap='gray')
        c3 = plt.colorbar(og, cmap='gray')
        c3.remove()
        anom_if = plt.imshow(anom_isofor, cmap=cmap2, vmax=1) #, alpha=.5
        c4 = plt.colorbar(anom_if, cmap=cmap2)
        c4.remove()
        plt.title(f'Iso For - Contam:{contamination}\nADS: {self.if_ad_score}% / ADE: {self.if_ad_eff}%')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        og = plt.imshow(self.data[:, :, 50], cmap='gray')
        c3 = plt.colorbar(og, cmap='gray')
        c3.remove()
        anom_rxs = plt.imshow(anom_lof_fp, cmap=cmap3, vmax=1)  # , alpha=.5
        c4 = plt.colorbar(anom_rxs, cmap=cmap3)
        c4.remove()
        plt.title(f'LOF-FP - C:{ncomps} / N:{neighbor}\nADS: {self.loffp_ad_score}% / ADE: {self.loffp_ad_eff}%')
        plt.axis('off')

        plt.subplot(1, 4, 4)
        og = plt.imshow(self.data[:, :, 50], cmap='gray')
        c5 = plt.colorbar(og, cmap='gray')
        c5.remove()
        anom_c1 = plt.imshow(anom_collection, cmap=cmap4) #, alpha=.5
        c6 = plt.colorbar(anom_c1, cmap=cmap4)
        c6.remove()
        plt.title(f'AD Stack 1\nADS: {self.as1b_ad_score}% / ADE: {self.as1b_ad_eff}%')
        plt.axis('off')
        plt.tight_layout(pad=1)

        if display:
            plt.show()

        if save:
            saveas = saveas = f'{save_to_location}/{self.file_name}-Stack1b.png'
            plt.savefig(saveas)
            plt.close()

    # Function to combine AD's: Filter, IF, RXSeg, LOF:F-P,
    def Anomaly_Stack_2(self, display=True, overlay=False, save=False):
        runtime_s2 = time_class('Stack 2')
        start_time = time.time()
        sc_var = 30
        cutoff_percent = (30, 100)
        contamination = 0.09
        seg_size = 250
        ncomps = 3
        neighbor = 1000
        bands = [4, 6, 10, 19, 28, 36, 46, 56, 76, 86, 96, 108, 120, 145, 180, 205, 235, 270]
        # bands = [0, 5, 10, 15, 25, 35, 45, 65, 85, 105, 135, 165, 195, 230, 270]
        # bands = [0, 5, 10, 20, 30, 50, 70, 100, 130, 170, 210, 250]
        # bands = [5, 10, 20, 30, 50, 80, 130, 190, 250]

        thread1 = threading.Thread(target=self.sc_filtered, args=(False, False, bands, sc_var, cutoff_percent))
        thread2 = threading.Thread(target=self.Anomaly_IF, args=(False, False, contamination))
        thread3 = threading.Thread(target=self.Anomaly_LOF_fp, args=(False, False, False, ncomps, neighbor))
        thread4 = threading.Thread(target=self.Anomaly_RXSeg, args=(False, False, seg_size))

        thread1.start(), thread2.start(), thread3.start(), thread4.start()
        thread1.join(), thread2.join(), thread3.join(), thread4.join()

        anom_subcat = self.anom_filter
        anom_isofor = self.anom_isofor
        anom_rx_seg = self.anom_rx_seg
        anom_lof_fp = self.anom_lof_fp

        anom_collection = np.zeros((self.data.shape[0], self.data.shape[1]), dtype=float)
        self.anom_stack_2 = np.zeros((self.data.shape[0], self.data.shape[1]), dtype=float)
        mask_one = mask_two = mask_three = np.zeros((self.data.shape[0], self.data.shape[1]), dtype=float)
        mask_four = mask_five = mask_six = np.zeros((self.data.shape[0], self.data.shape[1]), dtype=float)

        mask_one = np.logical_and(anom_subcat == 1, anom_isofor == 1)
        mask_two = np.logical_and(anom_subcat == 1, anom_rx_seg == 1)
        mask_three = np.logical_and(anom_subcat == 1, anom_lof_fp == 1)
        mask_four = np.logical_and(anom_rx_seg == 1, anom_isofor == 1)
        mask_five = np.logical_and(anom_rx_seg == 1, anom_lof_fp == 1)
        mask_six = np.logical_and(anom_lof_fp == 1, anom_isofor == 1)

        anom_collection[mask_one] = 1
        anom_collection[mask_two] = 1
        anom_collection[mask_three] = 1
        anom_collection[mask_four] = 1
        anom_collection[mask_five] = 1
        anom_collection[mask_six] = 1

        self.anom_stack_2[anom_collection == True] = 1
        self.anom_stack_2[anom_collection == False] = 0

        gt = ground_truth_class(self.raw_file_path)
        self.as2_ad_score, self.as2_ad_eff = gt.stats(self.anom_stack_1b)

        record_stack2 = record_class()
        record_stack2.record(
            filename=self.file_name,
            detector='Stack 2',
            score=self.as2_ad_score,
            eff=self.as2_ad_eff,
            runtime=runtime_s2.stats(),
            params={'Cat Var': sc_var, 'Cutoff': cutoff_percent, 'Contam': contamination, 'Seg Size': seg_size, 'Comps':ncomps, 'Neighbor': neighbor},
            imgstats=self.edit_record
        )

        cmap1 = cmap2 = cmap3 = cmap4 = cmap5 = 'Blues'

        if overlay:
            anom_subcat[anom_subcat == 0] = np.nan
            anom_isofor[anom_isofor == 0] = np.nan
            anom_rx_seg[anom_rx_seg == 0] = np.nan
            anom_lof_fp[anom_lof_fp == 0] = np.nan
            anom_collection[anom_collection == False] = np.nan
            cmap1 = cmap2 = cmap3 = cmap4 = cmap5 = 'brg'

        total_time = round(((time.time() - start_time) / 60), 1)
        print(f'Anomaly Stack Time: {total_time} mins')

        plt.figure(figsize=(15, 8))
        plt.suptitle(f'Anomaly Stack 2 Comparison - Time: {total_time} mins\n{self.file_name}')
        plt.subplot(1, 5, 1)
        og = plt.imshow(self.data[:, :, 50], cmap='gray')
        c1 = plt.colorbar(og, cmap='gray')
        c1.remove()
        anom_sc = plt.imshow(anom_subcat, cmap=cmap1, vmax=1)  # , alpha=.5
        c2 = plt.colorbar(anom_sc, cmap=cmap1)
        c2.remove()
        plt.title(f'Filtered - Var: {sc_var}/CO: {cutoff_percent}\nADS: {self.filter_ad_score}% / ADE: {self.filter_ad_eff}%')
        plt.axis('off')

        plt.subplot(1, 5, 2)
        og = plt.imshow(self.data[:, :, 50], cmap='gray')
        c3 = plt.colorbar(og, cmap='gray')
        c3.remove()
        anom_if = plt.imshow(anom_isofor, cmap=cmap2, vmax=1)  # , alpha=.5
        c4 = plt.colorbar(anom_if, cmap=cmap2)
        c4.remove()
        plt.title(f'Iso For - Contam: {contamination}\nADS: {self.if_ad_score}% / ADE: {self.if_ad_eff}%')
        plt.axis('off')

        plt.subplot(1, 5, 3)
        og = plt.imshow(self.data[:, :, 50], cmap='gray')
        c3 = plt.colorbar(og, cmap='gray')
        c3.remove()
        anom_rxs = plt.imshow(anom_rx_seg, cmap=cmap3, vmax=1)  # , alpha=.5
        c4 = plt.colorbar(anom_rxs, cmap=cmap3)
        c4.remove()
        plt.title(f'RX-Seg - Seg Size: {seg_size} Pixels\nADS: {self.rxs_ad_score}% / ADE: {self.rxs_ad_eff}%')
        plt.axis('off')

        plt.subplot(1, 5, 4)
        og = plt.imshow(self.data[:, :, 50], cmap='gray')
        c3 = plt.colorbar(og, cmap='gray')
        c3.remove()
        anom_loffp = plt.imshow(anom_lof_fp, cmap=cmap4, vmax=1)  # , alpha=.5
        c4 = plt.colorbar(anom_loffp, cmap=cmap4)
        c4.remove()
        plt.title(f'LOF: FP - PCA Comps: {self.pca_components}\nADS: {self.loffp_ad_score}% / ADE: {self.loffp_ad_eff}%')
        plt.axis('off')

        plt.subplot(1, 5, 5)
        og = plt.imshow(self.data[:, :, 50], cmap='gray')
        c5 = plt.colorbar(og, cmap='gray')
        c5.remove()
        anom_c1 = plt.imshow(anom_collection, cmap=cmap5)  # , alpha=.5
        c6 = plt.colorbar(anom_c1, cmap=cmap5)
        c6.remove()
        plt.title(f'AD Stack 2\nADS: {self.as2_ad_score}% / ADE: {self.as2_ad_eff}%')
        plt.axis('off')
        plt.tight_layout(pad=1)

        if display:
            plt.show()

        if save:
            saveas = (f'../../Dropbox/2 Work/1 Optics Lab/1 Anomaly Detection/Autosaves/{self.file_name}-AS')
            plt.savefig(saveas)
            plt.close()

    # Function to combine AD's: LOF-fp low comp and high comp and Iso For
    def Anomaly_Stack_3(self, display=True, overlay=False, save=False):
        runtime_s3 = time_class('Stack 3')
        start_time = time.time()

        comp_low = 3
        comp_high = 14
        contamination = 0.08
        neighbor = self.data.shape[2]

        image2 = Hyperspectral(self.raw_file_path)
        image2.preprocess(bands=(410, 900))

        image2.Anomaly_LOF_fp(False, False, False, comp_low, neighbor)
        self.Anomaly_LOF_fp(False, False, False, comp_high, neighbor)
        self.Anomaly_IF(False, False, contamination)

        # thread1 = threading.Thread(target=image2.Anomaly_LOF_fp, args=(False, True, False, comp_low, neighbor))
        # thread2 = threading.Thread(target=self.Anomaly_LOF_fp, args=(False, True, False, comp_high, neighbor))
        # thread3 = threading.Thread(target=self.Anomaly_IF, args=(False, False, contamination))
        #
        # thread1.start(), thread2.start(), thread3.start()
        # thread1.join(), thread2.join(), thread3.join()

        anom_lof_fp_low = image2.anom_lof_fp
        anom_lof_fp_high = self.anom_lof_fp
        anom_isofor = self.anom_isofor

        anom_collection = np.zeros((self.data.shape[0], self.data.shape[1]), dtype=float)
        self.anom_stack_lof = np.zeros((self.data.shape[0], self.data.shape[1]), dtype=float)
        mask_one = mask_two = mask_three = np.zeros((self.data.shape[0], self.data.shape[1]), dtype=float)

        mask_one = np.logical_and(anom_lof_fp_low, anom_lof_fp_high)
        mask_two = np.logical_and(anom_lof_fp_low, anom_isofor)
        mask_three = np.logical_and(anom_lof_fp_high, anom_isofor)

        anom_collection[mask_one] = 1
        anom_collection[mask_two] = 1
        anom_collection[mask_three] = 1

        self.anom_stack_lof[anom_collection == True] = 1
        self.anom_stack_lof[anom_collection == False] = 0

        gt = ground_truth_class(self.raw_file_path)
        self.aslof_ad_score, self.aslof_ad_eff = gt.stats(self.anom_stack_lof)

        record_stack3 = record_class()
        record_stack3.record(
            filename=self.file_name,
            detector='Stack 3',
            score=self.aslof_ad_score,
            eff=self.aslof_ad_eff,
            runtime=runtime_s3.stats(),
            params={'Comp Low': comp_low, 'Comp High': comp_high, 'Contam': contamination, 'Neighbor': neighbor},
            imgstats=self.edit_record
        )

        cmap1 = cmap2 = cmap3 = 'Blues'

        if overlay:
            anom_lof_fp_low[anom_lof_fp_low == 0] = np.nan
            anom_lof_fp_high[anom_lof_fp_high == 0] = np.nan
            anom_isofor[anom_isofor == 0] = np.nan
            anom_collection[anom_collection == 0] = np.nan
            cmap1 = cmap2 = cmap3 = 'brg'

        total_time = round(((time.time() - start_time) / 60), 1)
        print(f'Anomaly Stack LOF Time: {total_time} mins')

        plt.figure(figsize=(15, 8))
        plt.suptitle(f'Anomaly Stack 3 Comparison - Time: {total_time} mins\n{self.file_name}')

        plt.subplot(1, 4, 1)
        og = plt.imshow(self.data[:, :, 50], cmap='gray')
        c1 = plt.colorbar(og, cmap='gray')
        c1.remove()
        anom_sc = plt.imshow(anom_lof_fp_low, cmap=cmap1, vmax=1)  # , alpha=.5
        c2 = plt.colorbar(anom_sc, cmap=cmap1)
        c2.remove()
        plt.title(f'LOF: Low Comp-{comp_low} NN-{neighbor}\nADS: {image2.loffp_ad_score}% / ADE: {image2.loffp_ad_eff}%')
        plt.axis('off')

        plt.subplot(1, 4, 2)
        og = plt.imshow(self.data[:, :, 50], cmap='gray')
        c3 = plt.colorbar(og, cmap='gray')
        c3.remove()
        anom_if = plt.imshow(anom_lof_fp_high, cmap=cmap2, vmax=1)  # , alpha=.5
        c4 = plt.colorbar(anom_if, cmap=cmap2)
        c4.remove()
        plt.title(f'LOF: High Comp-{comp_high} NN-{neighbor}\nADS: {self.loffp_ad_score}% / ADE: {self.loffp_ad_eff}%')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        og = plt.imshow(self.data[:, :, 50], cmap='gray')
        c3 = plt.colorbar(og, cmap='gray')
        c3.remove()
        anom_if = plt.imshow(anom_isofor, cmap=cmap2, vmax=1)  # , alpha=.5
        c4 = plt.colorbar(anom_if, cmap=cmap2)
        c4.remove()
        plt.title(f'Iso For - Contam: {contamination}\nADS: {self.if_ad_score}% / ADE: {self.if_ad_eff}%')
        plt.axis('off')

        plt.subplot(1, 4, 4)
        og = plt.imshow(self.data[:, :, 50], cmap='gray')
        c3 = plt.colorbar(og, cmap='gray')
        c3.remove()
        anom_rxs = plt.imshow(anom_collection, cmap=cmap3, vmax=1)  # , alpha=.5
        c4 = plt.colorbar(anom_rxs, cmap=cmap3)
        c4.remove()
        plt.title(f'Anomaly Stack 3\nADS: {self.aslof_ad_score}% / ADE: {self.aslof_ad_eff}%')
        plt.axis('off')
        plt.tight_layout(pad=1)

        if display:
            plt.show()

        if save:
            saveas = f'{save_to_location}/Stack3-{self.file_name}.png'
            plt.savefig(saveas)
            plt.close()


#-----------------------------------------------------------------
# HYPERSPECTRAL AUXILIARY CLASSES
#-----------------------------------------------------------------
# ANOMALY DETECTOR CLASS
class anomaly_detector:

    def __init__(self, image):
        self.image = image

# VEGETATION INDICES CLASS
class vegetation_indices:

    def __init__(self, image):
        self.image = image

    # Function to display the NDVI
    def NDVI(self, display, save=False, r=650, n=850):
        print('NDVIing')
        switched_dict = {v: k for k, v in self.image.wavelengths_dict.items()}
        red, nir = None, None

        while red is None:
            red = switched_dict.get(r)
            r += 1
        while nir is None:
            nir = switched_dict.get(n)
            n += 1

        # print(f'red: {red} / r: {r} / nir: {nir} / n: {n}')
        RED = self.image.data[:, :, red]
        NIR = self.image.data[:, :, nir]

        RED, NIR = RED.astype('float'), NIR.astype('float')
        top, bottom = NIR - RED, NIR + RED
        top[top == 0], bottom[bottom == 0] = 0, np.nan

        ndvi_array = np.divide(top, bottom)
        ndvi_array[ndvi_array < 0] = 0
        ndvi_array[ndvi_array > 1] = 1

        self.index_ndvi = np.zeros((self.image.data.shape[0], self.image.data.shape[1]), dtype=float)
        self.index_ndvi = ndvi_array

        if save:
            plt.figure(figsize=(20, 25))
            plt.imshow(ndvi_array, cmap=plt.get_cmap("RdYlGn"))
            plt.title('NDVI')
            plt.colorbar()
            plt.axis('off')
            saveas = f'{save_to_location}/{self.image.file_name}-NDVI'
            plt.savefig(saveas)
            plt.close()

        if display:
            plt.imshow(ndvi_array, cmap=plt.get_cmap("RdYlGn"))
            plt.title('NDVI')
            plt.colorbar()
            plt.axis('off')
            plt.show()

# DATA LOGGING CLASS
class record_class:
    def __init__(self):

        self.record_location_base = 'Records/'
        self.record_location_ad = self.record_location_base + 'Anomaly Detectors/'

        # File Name, Detector, AD Score, AD Eff, Time, AD Parameters, Preprocesses, Image Stats
        self.fields = ['File Name', 'Detector', 'Score', 'Efficiency', 'Run Time', 'Parameters', 'Image Stats']

        if not os.path.isdir(self.record_location_ad):
            os.makedirs(self.record_location_ad)

    def record(self, filename, detector, score, eff, runtime, params, imgstats):
        import csv
        self.filename = self.record_location_ad + detector + '.csv'
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=self.fields)
                writer.writeheader()

        with open(self.filename, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fields)
            writer.writerow({self.fields[0] : filename,
                             self.fields[1] : detector,
                             self.fields[2] : score,
                             self.fields[3] : eff,
                             self.fields[4] : runtime,
                             self.fields[5] : params,
                             self.fields[6] : imgstats})

    def compile_detectors(self):

        try:

            if os.path.exists(f'{self.record_location_ad}Detector Data Main.csv'):
                os.remove(f'{self.record_location_ad}Detector Data Main.csv')

            # Get the list of all csv files in the folder
            csv_files = [f'{self.record_location_ad}{f}' for f in os.listdir(self.record_location_ad) if f.endswith('.csv')]

            # Create an empty DataFrame to store the data from all csv files
            df = pd.DataFrame(
                columns=['File Name', 'Detector', 'Score', 'Efficiency', 'Run Time', 'Parameters', 'Image Stats'])

            # Loop through all csv files in the folder
            for file in csv_files:
                # Read the csv file into a DataFrame
                file_df = pd.read_csv(file)

                # Extract the specified columns from the csv file
                data = file_df[['File Name', 'Detector', 'Score', 'Efficiency', 'Run Time', 'Parameters', 'Image Stats']]

                # Append the data from the csv file to the DataFrame
                df = df.append(data)

            # Write the DataFrame to a new csv file
            df.to_csv(f'{self.record_location_ad}Detector Data Main.csv', index=False)
        except:
            print('DETECTOR DATA MAIN COULD NOT BE COMPILED. CHECK FILES.')

# TIME CLASS TO GIVE STATS ABOUT HOW LONG FUNCTION TAKES
class time_class:
    def __init__(self, name):
        self.start_time = time.time()
        self.name = name

    def stats(self):
        total_time = round((time.time() - self.start_time), 1)

        if total_time < 60:
            print(f'{self.name} Time: {total_time} secs')
        else:
            total_time_min = round((total_time / 60), 1)
            print(f'{self.name} Time: {total_time_min} mins')

        return total_time

# GROUND TRUTH CLASS TO COMPARE ANOMALY DETECTION
class ground_truth_class:

    def __init__(self, filepath):
        self.filepath = filepath

    # Function to check if there is a ground truth
    def check_if_gt(self, filepath):
        self.gt_file_path = filepath + '.png'
        # print(self.gt_file_path)

        try:
            iio.imread(self.gt_file_path)
            return True
        except: return False

    # Function to resize the ground truth to the array size
    def resize(self, y, x, stats=False):
        from skimage.transform import resize

        if self.check_if_gt(self.filepath):
            # Get File Type
            try:
                file = self.gt_file_path.split('.')
                self.file_type = file[1]

                try:
                    file_name = file[0].split('/')
                    self.file_name = file_name[-1]
                    # print(file_name[-1])
                except:
                    self.file_name = file[0]
            except:
                print('No File Type')

            ground_truth = np.zeros((y, x), dtype=float)
            self.data = iio.imread(self.gt_file_path)
            data = self.data
            gt = resize(data, ground_truth.shape, anti_aliasing=False)
            gt = gt[:,:, 0]
            gt[gt > 0] = 1
            gt = gt.astype(np.float32)
            gt = 1 - gt
            self.ground_truth = gt

        else:
            ground_truth = np.zeros((y, x), dtype=float)
            gt = resize(ground_truth, ground_truth.shape, anti_aliasing=False)
            gt = gt.astype(np.float32)
            self.ground_truth = gt

        if stats:
            print(f'OG GT Shape: {ground_truth.shape}')
            print(f'OG Data Shape: {self.data.shape}')
            print(f'Resized Data Shape: {gt.shape}')
            print(f'GT Shape: {gt.shape}')
            print(np.max(gt))

    # Function to display the Image
    def display_gt(self, overlay):
        gt = self.ground_truth

        if overlay:
            gt[gt == 0] = np.nan

        plt.imshow(gt, cmap=plt.get_cmap("RdYlGn"), vmax=1)
        plt.axis('off')
        plt.show()

    # Function to create stats comparing GT to AD
    def stats(self, matrix, stats=False):

        self.resize(matrix.shape[0], matrix.shape[1], stats=False)

        total_gt_pixels = np.count_nonzero(self.ground_truth) # how many pixels are in ground truth matrix
        total_matrix_pixels = np.count_nonzero(matrix)  # how many pixels anomaly detector found
        total_pixels = matrix.shape[0] * matrix.shape[1] # how many pixels in matrix
        x = np.logical_and(matrix, self.ground_truth)
        anom_num = np.count_nonzero(x)

        if total_gt_pixels == 0:
            if total_matrix_pixels == 0:
                anom_score = 100
                anom_eff = 100
            else:
                anom_score = round(((1 - (total_matrix_pixels / total_pixels)) * 100), 1)
                anom_eff = round(((1 - (total_matrix_pixels / total_pixels)) * 100), 1)
        else:
            anom_score = round(((anom_num / total_gt_pixels) * 100), 1)
            anom_eff = round(((anom_num / total_matrix_pixels) * 100), 1)

        # print(f'Anom Score: {anom_score}% / Noise Score: {noise_score}%')

        if stats:
            print(f'Total Pixels: {total_pixels}')
            print(f'Total Matrix Pixels: {total_matrix_pixels}')
            print(f'Total GT Pixels: {total_gt_pixels}')
            print(f'Matrix & Ground Truth: {anom_num}')
            # print(f'Pixels: {total_gt_pixels}')
            # print(f'Pixels: {total_gt_pixels}')
            # print(f'Pixels: {total_gt_pixels}')
            # print(f'Pixels: {total_gt_pixels}')



        return anom_score, anom_eff

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

# MATERIAL SYNTHESIS CLASS TO ADD MATERIALS IN DIFFERENT WAYS TO IMAGE
class material_synthesis:

    def __init__(self, filepath):
        self.image = Hyperspectral(filepath, stats=False)

    # Function to add anomaly using material & image as is
    def add_anomaly_1(self, material, location, size, scale_factor):
        mat = material.map_material_to_image(list(self.image.wavelengths_dict.values()))
        x_list = [(location[0] - size), (location[0] + size)]
        y_list = [location[1] + size, location[1] - size]
        adjusted = [(x * scale_factor) for x in mat]

        # Add the anomaly in the right shape
        for i in range(0, len(mat)):
            self.image.data[y_list[1]:y_list[0], x_list[0]:x_list[1], i] = adjusted[i]

        record = 'Anomaly Synthesis: anomaly added = {}, method = 1, size = {}, location = {}, scale_factor = {}'.format(
            material.material_id, size, location, scale_factor)
        self.image.edit_record.append(record)

    # Function to add anomaly using material spectral mapping to image using image pixel
    def add_anomaly_2(self, material, location, size):
        mat = material.map_material_to_image(list(self.image.wavelengths_dict.values()))
        x_list = [(location[0] - size), (location[0] + size)]
        y_list = [location[1] + size, location[1] - size]
        min_max = self.image.pixel_metadata(location[0], location[1])
        ad = np.array(mat).reshape(-1, 1)
        scaler = preprocessing.MinMaxScaler(feature_range=(min_max[0], min_max[1]))
        normalizedlist = scaler.fit_transform(ad)
        adjusted = [round(i[0]) for i in normalizedlist]

        # Add the anomaly in the right shape
        for i in range(0, len(mat)):
            self.image.data[y_list[1]:y_list[0], x_list[0]:x_list[1], i] = adjusted[i]

        record = 'Anomaly Synthesis: anomaly added = {}, method = 2, size = {}, location = {}'.format(
            material.material_id, size, location)
        self.image.edit_record.append(record)

    # Function to add anomaly using material spectral mapping to image using image area
    def add_anomaly_3(self, material, location, size):
        mat = material.map_material_to_image(list(self.image.wavelengths_dict.values()))
        x_list = [(location[0] - size), (location[0] + size)]
        y_list = [location[1] + size, location[1] - size]
        min_max = self.image.area_metadata(y_list[1], y_list[0], x_list[0], x_list[1])
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
            self.image.data[y_list[1]:y_list[0], x_list[0]:x_list[1], i] = adjusted[i]

        record = 'Anomaly Synthesis: anomaly added = {}, method = 3, size = {}, location = {}'.format(
            material.material_id, size, location)
        self.image.edit_record.append(record)

    # Function to add anomaly using image's min/max values for material's % reflectivity
    def add_anomaly_4(self, material, location, size):
        mat = material.map_material_to_image(list(self.image.wavelengths_dict.values()))
        x_list = [(location[0] - size), (location[0] + size)]
        y_list = [location[1] + size, location[1] - size]
        # min_max = self.image_metadata()
        min_max = [-43, 9155, 840.4]
        # min_max = [-50, 32767, 983.8]
        mat_ar = np.array(mat)
        mat_max = mat_ar.max()
        map_max = (mat_max / 100) * min_max[1]
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
            self.image.data[y_list[1]:y_list[0], x_list[0]:x_list[1], i] = adjusted[i]

        record = 'Anomaly Synthesis: anomaly added = {}, method = 4, size = {}, location = {}'.format(
            material.material_id, size, location)
        self.image.edit_record.append(record)

    # Function to add anomaly A2 with noise added
    def add_anomaly_5(self, material, location, size, variation):
        mat = material.map_material_to_image(list(self.image.wavelengths_dict.values()))
        x_list = [(location[0] - size), (location[0] + size)]
        y_list = [location[1] + size, location[1] - size]
        min_max = self.image.pixel_metadata(location[0], location[1])

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
                    self.image.data[i, j, k] = adjusted[k]

                adjusted = []

        record = 'Anomaly Synthesis: anomaly added = {}, method = 5, size = {}, location = {}, variation = {}'.format(
            material.material_id, size, location, variation)
        self.image.edit_record.append(record)

    # Function to add anomaly like A5 but with blurred edges
    def add_anomaly_7(self, material, location, size, variation, weight):
        mat = material.map_material_to_image(list(self.image.wavelengths_dict.values()))
        x_list = [(location[0] - size), (location[0] + size)]
        y_list = [location[1] + size, location[1] - size]
        min_max = self.image.pixel_metadata(location[0], location[1])

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
                    self.image.data[i, j, k] = adjusted[k]

                adjusted = []
        record = 'Anomaly Synthesis: anomaly added = {}, method = 7, size = {}, location = {}, variation = {}, weight = {}'.format(
            material.material_id, size, location, variation, weight)
        self.image.edit_record.append(record)

        for i in range(0, len(mat)):
            # Top Edge: E1
            y1 = y_list[1] - 1
            a = self.image.data[y1, x_list[0]:x_list[1], i]  # surround pixels
            b = self.image.data[y_list[1], x_list[0]:x_list[1], i]  # material edge
            w1 = 1
            for x, y in zip(a, b):
                if x > y:
                    w1 = 1 + weight
                else:
                    w1 = 1 - weight
            q = ((w1 * a) + b) / 2
            self.image.data[y_list[1], x_list[0]:x_list[1], i] = q

            # Bottom Edge: E2
            y2 = y_list[0] - 1
            c = self.image.data[y2, x_list[0]:x_list[1], i]  # surround pixels
            d = self.image.data[y_list[0], x_list[0]:x_list[1], i]  # material edge
            w2 = 1
            for x, y in zip(c, d):
                if x > y:
                    w2 = 1 + weight
                else:
                    w2 = 1 - weight
            r = ((w2 * c) + d) / 2
            self.image.data[y_list[0], x_list[0]:x_list[1], i] = r

            # Left Edge: E3
            x1 = x_list[0] - 1
            y1 = y_list[1] + 1
            y2 = y_list[0] + 1
            e = self.image.data[y1: y2, x1, i]  # surround pixels
            f = self.image.data[y1: y2, x_list[0], i]  # material edge
            w3 = 1
            for x, y in zip(e, f):
                if x > y:
                    w3 = 1 + weight
                else:
                    w3 = 1 - weight
            s = ((w3 * e) + f) / 2
            self.image.data[y1: y2, x_list[0], i] = s

            # Right Edge: E4
            x2 = x_list[1] - 1
            g = self.image.data[y1: y2, x2, i]  # surround pixels
            h = self.image.data[y1: y2, x_list[1], i]  # material edge
            w4 = 1
            for x, y in zip(g, h):
                if x > y:
                    w4 = 1 + weight
                else:
                    w4 = 1 - weight
            t = ((w4 * g) + h) / 2
            self.image.data[y1: y2, x_list[1], i] = t

        # self.anom_record.append('A5_{}'.format(material.material_id))

    # Function to add anomaly like A5 but to better map the material's profile to images by averaging
    def add_anomaly_8(self, material, location, size, variation, weight):
        mat = material.map_material_to_image(list(self.image.wavelengths_dict.values()))
        x_list = [(location[0] - size), (location[0] + size)]
        y_list = [location[1] + size, location[1] - size]
        min_max = self.image.pixel_metadata(location[0], location[1])

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
                    a = self.image.data[i, j, k]  # OG Value for the pixel about to replace
                    b = adjusted[k]  # adjusted material value
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
                    self.image.data[i, j, k] = q

                adjusted = []

        record = 'Anomaly Synthesis: anomaly added = {}, method = 8, size = {}, location = {}, variation = {}, weight = {}'.format(
            material.material_id, size, location, variation, weight)
        self.image.edit_record.append(record)

    # Function to create new material text file
    def create_material(self, location, filename, material_info):
        # check to see if file already exists

        values_single = []
        for i in range(224):
            values_single.append(self.image.data[location[1], location[0], i])
        g = open('Materials/Material Data/' + filename, 'w')
        for i in range(len(m.infonames)):
            d = m.infonames[i], ': ', material_info[i], '\n'
            g.writelines(d)
        g.writelines('\n')
        wave = list(self.image.wavelengths_dict.values())
        for i in range(len(self.image.wavelengths)):
            d = str((wave[i] / 1000)), ' ', str(values_single[i]), '\n'
            g.writelines(d)
        g.close()

#-----------------------------------------------------------------
# MAPIR RELATED FUNCTIONS
#-----------------------------------------------------------------
# Function to graph all files from Monochromator
def mapir_graph_all():
    pika_f = np.arange(500, 880, 5)
    for band in pika_f:
        try:
            file = f'RGN Files/MonoChrom Test 4/pika/radiance/{band}nm.bip'
            image = Hyperspectral(file)  # a hyperspectral image object using hyperspectral
            image.reduce_bands(bands=(490,880))
            image.graph_all_mapir_pika(display=False, save=False)
        except:
            continue
    plt.title(f'MC Test: Pika')
    plt.show()

# Function to graph all files from Monochromator
def mapir_graph_mapir():
    pika_f = np.arange(525, 576, 5)
    for band in pika_f:
        file = f'RGN Files/MonoChrom Test 4/pika/radiance/{band}nm.bip'
        image = Hyperspectral(file)  # a hyperspectral image object using hyperspectral
        image.reduce_bands(bands=(490,880))
        image.graph_all_mapir_pika(display=False, save=False)

    pika_f = np.arange(635, 686, 5)
    for band in pika_f:
        file = f'RGN Files/MonoChrom Test 4/pika/radiance/{band}nm.bip'
        image = Hyperspectral(file)  # a hyperspectral image object using hyperspectral
        image.reduce_bands(bands=(490,880))
        image.graph_all_mapir_pika(display=False, save=False)

    pika_f = np.arange(810, 871, 5)
    for band in pika_f:
        file = f'RGN Files/MonoChrom Test 4/pika/radiance/{band}nm.bip'
        image = Hyperspectral(file)  # a hyperspectral image object using hyperspectral
        image.reduce_bands(bands=(490,880))
        image.graph_all_mapir_pika(display=False, save=False)

    plt.title(f'MC Test: Pika like MapIR')
    plt.show()

# Function to graph all files from Monochromator
def pika_to_mapir():
    red = []
    firstr = True
    pika_f = np.arange(520, 581, 5)
    for band in pika_f:
        file = f'RGN Files/MonoChrom Test 4/pika/radiance/{band}nm.bip'
        image = Hyperspectral(file)  # a hyperspectral image object using hyperspectral
        image.reduce_bands(bands=(500,900))
        max_arrayr = image.max_array()
        if firstr:
            red = max_arrayr
            firstr = False
        else:
            for i in range(len(red)):
                red[i] = red[i] + max_arrayr[i]


    green = []
    firstg = True
    pika_f = np.arange(630, 691, 5)
    for band in pika_f:
        file = f'RGN Files/MonoChrom Test 4/pika/radiance/{band}nm.bip'
        image = Hyperspectral(file)  # a hyperspectral image object using hyperspectral
        image.reduce_bands(bands=(500,900))
        max_arrayg = image.max_array()
        if firstg:
            green = max_arrayg
            firstg = False
        else:
            for i in range(len(green)):
                green[i] = green[i] + max_arrayg[i]

    nir = []
    firstn = True
    pika_f = np.arange(810, 871, 5)
    for band in pika_f:
        file = f'RGN Files/MonoChrom Test 4/pika/radiance/{band}nm.bip'
        image = Hyperspectral(file)  # a hyperspectral image object using hyperspectral
        image.reduce_bands(bands=(500,900))
        max_arrayn = image.max_array()
        if firstn:
            nir = max_arrayn
            firstn = False
        else:
            for i in range(len(nir)):
                nir[i] = nir[i] + max_arrayn[i]

    x_values = np.linspace(500, 900, len(red))
    plt.plot(x_values, green, color='r', linewidth=2, label='Red Values')
    plt.plot(x_values, red, color='g', linewidth=2, label='Green Values')
    plt.plot(x_values, nir,color='b', linewidth=2, label='NIR Values')
    plt.vlines(x=[550, 650, 850], ymin=0, ymax=255, colors='black', ls='--', lw=1, label='MapIR Bands')
    plt.xlabel('Bands')
    plt.ylabel('Counts')
    # plt.xticks([x for x in range(500, 900, 25)])
    plt.title(f'MC Test: Pika like MapIR')
    plt.legend(loc='upper left')
    plt.show()

# Function to integrate the response for each band for calibration using numpy
def integrate_np(display, stats, prnt):

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

    green_list = []
    firstg = True
    pika_f = np.arange(520, 581, 5)
    for band in pika_f:
        file = f'RGN Files/MonoChrom Test 4/pika/radiance/{band}nm.bip'
        image = Hyperspectral(file)  # a hyperspectral image object using hyperspectral
        image.reduce_bands(bands=(500,900))
        max_arrayg = image.max_array()
        if firstg:
            green_list = max_arrayg
            firstg = False
        else:
            for i in range(len(green_list)):
                green_list[i] = green_list[i] + max_arrayg[i]


    red_list = []
    firstr = True
    pika_f = np.arange(630, 691, 5)
    for band in pika_f:
        file = f'RGN Files/MonoChrom Test 4/pika/radiance/{band}nm.bip'
        image = Hyperspectral(file)  # a hyperspectral image object using hyperspectral
        image.reduce_bands(bands=(500,900))
        max_arrayr = image.max_array()
        if firstr:
            red_list = max_arrayr
            firstr = False
        else:
            for i in range(len(red_list)):
                red_list[i] = red_list[i] + max_arrayr[i]

    nir_list = []
    firstn = True
    pika_f = np.arange(810, 871, 5)
    for band in pika_f:
        file = f'RGN Files/MonoChrom Test 4/pika/radiance/{band}nm.bip'
        image = Hyperspectral(file)  # a hyperspectral image object using hyperspectral
        image.reduce_bands(bands=(500,900))
        max_arrayn = image.max_array()
        if firstn:
            nir_list = max_arrayn
            firstn = False
        else:
            for i in range(len(nir_list)):
                nir_list[i] = nir_list[i] + max_arrayn[i]

    if stats:
        print(green_list)
        print(red_list)
        print(nir_list)
    # Integration values from numpy
    #500-600nm
    yr1 = [red_list[8:44]]
    yg1 = [green_list[8:44]]
    yn1 = [nir_list[8:44]]
    ga1 = np.trapz(yr1)
    ga2 = np.trapz(yg1)
    ga3 = np.trapz(yn1)

    # 600-700nm
    yr2 = [0, *red_list[60:98], 0]
    yg2 = [0, *green_list[60:98], 0]
    yn2 = [0, *nir_list[60:98], 0]
    ra1 = np.trapz(yr2)
    ra2 = np.trapz(yg2)
    ra3 = np.trapz(yn2)

    # 700-850nm
    yr3 = [0, *red_list[106:180], 0]
    yg3 = [0, *green_list[106:180], 0]
    yn3 = [0, *nir_list[106:180], 0]
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

    print(f'R{rsum} / G{gsum} / N{nsum}')

    Ra1, Ra2, Ra3 = round((red[0] / rsum), 2), round((red[1] / gsum), 2), round((red[2] / nsum), 2)
    Ga1, Ga2, Ga3 = round((green[0] / rsum), 2), round((green[1] / gsum), 2), round((green[2] / nsum), 2)
    Na1, Na2, Na3 = round((nir[0] / rsum), 2), round((nir[1] / gsum), 2), round((nir[2] / nsum), 2)
    calibration = [[Ra1, Ra2, Ra3], [Ga1, Ga2, Ga3], [Na1, Na2, Na3]]

    if stats:
        print(f'RED: {red}')
        print('-' * 30)
        print(f'GREEN: {green}')
        print('-' * 30)
        print(f'NIR: {nir}')
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

# Function to graph all files from Monochromator
def mapir_cor_max(display, save):
    pika_f = np.arange(400, 880, 10)
    max_dict = {}
    for band in pika_f:
        file = f'RGN Files/PikaC T3/{band}nm-Radiance From Raw Data.bip'
        image = Hyperspectral(file)  # a hyperspectral image object using hyperspectral
        list = image.image_metadata()
        max_dict.update( {list[1]:band } )

    if display:
        plt.scatter(max_dict.values(),max_dict.keys())
        plt.title('Pika-Corrected: Max Values')
        plt.xlabel('Bands')
        plt.ylabel('Values')
        plt.show()

    if save:
        saveas = (f'../../Dropbox/2 Work/1 Optics Lab/2 Projects/MapIR/Autosaves/Pika-Corrected Max Values Dict.txt')
        with open(saveas, 'w') as f:
            for x,y in max_dict.items():
                line = f'{str(y)} : {str(x)}\n'
                f.write(line)
            f.close()

# Function to graph all files from Monochromator
def mapir_cor_av(display, save):
    pika_f = np.arange(400, 880, 10)
    max_dict = {}
    for band in pika_f:
        file = f'RGN Files/PikaC T3/{band}nm-Radiance From Raw Data.bip'
        image = Hyperspectral(file)  # a hyperspectral image object using hyperspectral
        list = image.image_metadata()
        max_dict.update( {list[2]:band } )

    if display:
        plt.scatter(max_dict.values(),max_dict.keys())
        plt.title('Pika-Corrected: Mean Values')
        plt.xlabel('Bands')
        plt.ylabel('Values')
        plt.show()

    if save:
        saveas = (f'../../Dropbox/2 Work/1 Optics Lab/2 Projects/MapIR/Autosaves/Pika-Corrected Mean Values Dict.txt')
        with open(saveas, 'w') as f:
            for x,y in max_dict.items():
                line = f'{str(y)} : {str(x)}\n'
                f.write(line)
            f.close()

# Function to compare the wavelength from monocrhom and pika
def mono_pika_comp():

    folder_path = 'RGN Files/MonoChrom Test 4/pika/radiance'
    diff = []
    # loop through all files in the folder
    for filename in os.listdir(folder_path):
        # get the full path of the file
        file_path = os.path.join(folder_path, filename)
        # check if the file is a regular file
        if os.path.isfile(file_path):
            if filename[-1] != 'r':
            # do something with the file
                image = Hyperspectral(file_path)
                diff.append(image.mono_pika_comp(p=False))
        else:
            print('No File')

    min = np.min(diff)
    max = np.max(diff)
    av = round(np.mean(diff), 1)
    print(f'Min: {min} / Max: {max} / Average: {av}')

# Function to return amounts to adjust wavelength dict
def wavelength_correct():

    folder_path = 'Datasets/RGN Files/MonoChrom Test 4/pika/radiance'
    diff = []
    # loop through all files in the folder
    for filename in os.listdir(folder_path):
        # get the full path of the file
        file_path = os.path.join(folder_path, filename)
        # check if the file is a regular file
        if os.path.isfile(file_path):
            if filename[-1] != 'r':
            # do something with the file
                image = Hyperspectral(file_path)
                diff.append(image.mono_pika_comp(p=False))
        else:
            print('No File')

    return diff


#-------------------------------------------------------------------------------
# PIPELINE FUNCTIONS
# -------------------------------------------------------------------------------
# Function to Rename Data Files
def data_process(filepath):
    pass

# Function to preprocess all images and export
def preprocess(dataset):
    i = 1
    for file in dataset:
        print(f'------------------ Cube {i} ------------------')
        runtime_preprocess = time_class('Preprocess & Export')
        image = Hyperspectral(file, stats=False)
        image.smooth_average()
        image.reduce_bands(bands=(410, 900))
        image.check_values()
        image.export('Processed')
        runtime_preprocess.stats()
        i += 1

# Function to generate overview files for all files
def overview_all(dataset):
    i = 1
    for file in dataset:
        print(f'------------------ Cube {i} ------------------')
        image = Hyperspectral(file, stats=False)
        image.overview(display=False, save=True)
        i += 1

# Function for cropping
def crop_export(file, bounds, iteration):
    image = Hyperspectral(file, stats=False)
    image.crop(bounds)
    image.export(iteration)