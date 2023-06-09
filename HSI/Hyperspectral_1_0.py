#Hyperspectral is used to create HSI objects and functions
#Kevin McKenzie 2022

from Materials.Materials import Material as m
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.stats import chi2
from copy import deepcopy
import concurrent.futures
from Hyp_envi import Cube
import imageio.v3 as iio
from PIL import Image
import pandas as pd
import numpy as np
import scipy.io
import spectral
import random
import time
import math
import os

# with concurrent.futures.ProcessPoolExecutor() as executor:

AV_Full = 'Datasets/AVIRIS/f100704t01p00r06rdn_b_sc01_ort_img'
Pika_C = 'Datasets/Pika/c_mosaic.bil'
Pika_D = 'Datasets/Pika/d_mosaic.bil'
AV_Crop_3A_L = 'Anomaly Files/AV-Crop-3Anom-L'
AV_Crop_3A_M = 'Anomaly Files/AV-Crop-3Anom-M'
AV_Crop_7A = 'Anomaly Files/AV-Crop-7A'
AV_Crop_8A = 'Anomaly Files/AV-Crop2-8A'

files = ['Anomaly Files/AV-Crop',
         'Anomaly Files/AV-Crop2',
         'Anomaly Files/AV-Crop-3Anom-S',
         'Anomaly Files/AV-Crop2-9A',
         'Datasets/Botswana/Botswana.mat',
         'Datasets/Kennedy Space/KSC.mat',
         'Datasets/Indian Pines/Indian_pines.mat',
         'Datasets/Pavia/Pavia.mat',
         'Datasets/Pavia/PaviaU.mat',
         'Datasets/Salinas/Salinas.mat',
         'Datasets/ABU/abu-airport-1.mat', 'Datasets/ABU/abu-airport-2.mat', 'Datasets/ABU/abu-airport-3.mat',
         'Datasets/ABU/abu-beach-1.mat', 'Datasets/ABU/abu-beach-2.mat', 'Datasets/ABU/abu-beach-3.mat',
         'Datasets/ABU/abu-urban-1.mat', 'Datasets/ABU/abu-urban-2.mat', 'Datasets/ABU/abu-urban-3.mat',
         'Datasets/ABU/abu-urban-4.mat', 'Datasets/ABU/abu-urban-5.mat' ]
files_dict = {'AV Crop 1' :'Anomaly Files/AV-Crop',
                'AV Crop 2' : 'Anomaly Files/AV-Crop2',
                'AV Crop 1: 3A S' : 'Anomaly Files/AV-Crop-3Anom-S',
                'AV Crop 2: 9A' : 'Anomaly Files/AV-Crop2-9A',
                'Botswana' : 'Datasets/Botswana/Botswana.mat',
                'Kennedy Space' : 'Datasets/Kennedy Space/KSC.mat',
                'Indian Pines' : 'Datasets/Indian Pines/Indian_pines.mat',
                'Pavia' : 'Datasets/Pavia/Pavia.mat',
                'PaviaU' : 'Datasets/Pavia/PaviaU.mat',
                'Salinas' : 'Datasets/Salinas/Salinas.mat',
                'ABU Airport 1' : 'Datasets/ABU/abu-airport-1.mat',
                'ABU Airport 2' : 'Datasets/ABU/abu-airport-2.mat',
                'ABU Airport 3' : 'Datasets/ABU/abu-airport-3.mat',
                'ABU Beach 1' : 'Datasets/ABU/abu-beach-1.mat',
                'ABU Beach 2' : 'Datasets/ABU/abu-beach-2.mat',
                'ABU Beach 3' : 'Datasets/ABU/abu-beach-3.mat',
                'ABU Urban 1' : 'Datasets/ABU/abu-urban-1.mat',
                'ABU Urban 2' : 'Datasets/ABU/abu-urban-2.mat',
                'ABU Urban 3' : 'Datasets/ABU/abu-urban-3.mat',
                'ABU Urban 4' : 'Datasets/ABU/abu-urban-4.mat',
                'ABU Urban 5' : 'Datasets/ABU/abu-urban-5.mat' }
avc_set = ['Datasets/AVC Set/AVC-1', 'Datasets/AVC Set/AVC-2',
           'Datasets/AVC Set/AVC-3', 'Datasets/AVC Set/AVC-4',
           'Datasets/AVC Set/AVC-5', 'Datasets/AVC Set/AVC-6',
           'Datasets/AVC Set/AVC-7', 'Datasets/AVC Set/AVC-8',
           'Datasets/AVC Set/AVC-9', 'Datasets/AVC Set/AVC-10',
           'Datasets/AVC Set/AVC-11', 'Datasets/AVC Set/AVC-12',
           'Datasets/AVC Set/AVC-13', 'Datasets/AVC Set/AVC-14',
           'Datasets/AVC Set/AVC-15', 'Datasets/AVC Set/AVC-16']

def show_all_files():
    print('0: AV-Crop')
    print('1: AV-Crop2')
    print('2: AV-Crop-3Anom-S')
    print('3: AV-Crop2-9A')
    print('4: Botswana')
    print('5: KSC')
    print('6: Indian_pines')
    print('7: Pavia')
    print('8: PaviaU')
    print('9: Salinas')
    print('10: ABU-Airport 1')
    print('11: ABU-Airport 2')
    print('12: ABU-Airport 3')
    print('13: ABU-Beach 1')
    print('14: ABU-Beach 2')
    print('15: ABU-Beach 3')
    print('16: ABU-Urban 1')
    print('17: ABU-Urban 2')
    print('18: ABU-Urban 3')
    print('19: ABU-Urban 4')
    print('20: ABU-Urban 5')

# Function to compare all files in a set
def compare_all_files(file):

    for f in file:
        image = Hyperspectral(f)
        image = image.reduce_bands()
        image.compare(display=False, save=True)

# Function to graph all files from Monochromator
def mapir_graph_all():
    pika_f = np.arange(500, 880, 10)
    for band in pika_f:
        file = f'RGN Files/MonoChrom Test 4/pika/radiance/{band}nm.bip'
        image = Hyperspectral(file)  # a hyperspectral image object using hyperspectral
        image.graph_mapir_pika(display=True, save=False)

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
    def __init__(self, raw_file_path):

        self.edit_record = []
        self.file_path = raw_file_path

        try:
            file = raw_file_path.split('.')
            self.file_type = file[1]

            try:
                file_name = file[0].split('/')
                self.file_name = file_name[-1]
                print(file_name[-1])
            except:
                pass
        except:
            self.file_type = 'envi'
            try:
                file_name = raw_file_path.split('/')
                self.file_name = file_name[-1]
                print(file_name[-1])
            except:
                pass

        if self.file_type == 'bil':
            data = Cube.from_path(self.file_path)
            data = data.read()  # load into memory in native format as numpy array
            # or load as memory map by adding as_mmap=True
            # then only the parts of the cube you access will be read into memory
            # if you need just a small part this can be much faster
            # print(data)
            data = np.array(data).astype(np.float32)

            self.img_x = int(data.shape[2])
            self.img_y = int(data.shape[0])
            self.img_bands = int(data.shape[1])

            # reshape_data = np.zeros((self.img_y, self.img_x, self.img_bands), dtype=float)
            self.data = data.transpose((0,2,1))
            self.open_HDR()

        if self.file_type == 'bip':
            data = Cube.from_path(self.file_path)
            data = data.read()  # load into memory in native format as numpy array
            # or load as memory map by adding as_mmap=True
            # then only the parts of the cube you access will be read into memory
            # if you need just a small part this can be much faster
            # print(data)
            data = np.array(data).astype(np.float32)

            self.img_x = int(data.shape[1])
            self.img_y = int(data.shape[0])
            self.img_bands = int(data.shape[2])

            # reshape_data = np.zeros((self.img_y, self.img_x, self.img_bands), dtype=float)
            self.data = data.transpose((0,1,2))
            self.open_HDR()

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

    # -------------------------------------------------------------------------------
    # EDITING FUNCTIONS
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

        image = self.file_path.split('/')
        image = image[-1]

        record = 'Image Metadata: image = {}, min = {}, max = {}, average = {}'.format(
            image, min_total, max_total, av_total)
        self.edit_record.append(record)

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

        if self.file_type == 'envi':
            im_crop.header_file_dict['samples'] = str(im_crop.img_x)
            im_crop.header_file_dict['lines'] = str(im_crop.img_y)

        image = self.file_path.split('/')
        image = image[-1]

        record = 'Image Edit: edit = crop, image = {}, dimensions = {}'.format(
            image, dimension)
        self.edit_record.append(record)

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

    # Function to add anomaly using material & image as is
    def add_anomaly_1(self, material, location, size, scale_factor):
        mat = material.map_material_to_image(list(self.wavelengths_dict.values()))
        x_list = [(location[0] - size), (location[0] + size)]
        y_list = [location[1] + size, location[1] - size]
        adjusted = [(x * scale_factor) for x in mat]

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
        min_max = self.pixel_metadata(location[0], location[1])
        ad = np.array(mat).reshape(-1, 1)
        scaler = preprocessing.MinMaxScaler(feature_range=(min_max[0], min_max[1]))
        normalizedlist = scaler.fit_transform(ad)
        adjusted = [round(i[0]) for i in normalizedlist]

        # Add the anomaly in the right shape
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
            a = self.data[y1, x_list[0]:x_list[1], i]  # surround pixels
            b = self.data[y_list[1], x_list[0]:x_list[1], i]  # material edge
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
            e = self.data[y1: y2, x1, i]  # surround pixels
            f = self.data[y1: y2, x_list[0], i]  # material edge
            w3 = 1
            for x, y in zip(e, f):
                if x > y:
                    w3 = 1 + weight
                else:
                    w3 = 1 - weight
            s = ((w3 * e) + f) / 2
            self.data[y1: y2, x_list[0], i] = s

            # Right Edge: E4
            x2 = x_list[1] - 1
            g = self.data[y1: y2, x2, i]  # surround pixels
            h = self.data[y1: y2, x_list[1], i]  # material edge
            w4 = 1
            for x, y in zip(g, h):
                if x > y:
                    w4 = 1 + weight
                else:
                    w4 = 1 - weight
            t = ((w4 * g) + h) / 2
            self.data[y1: y2, x_list[1], i] = t

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
                    a = self.data[i, j, k]  # OG Value for the pixel about to replace
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
                    self.data[i, j, k] = q

                adjusted = []

        record = 'Anomaly Synthesis: anomaly added = {}, method = 8, size = {}, location = {}, variation = {}, weight = {}'.format(
            material.material_id, size, location, variation, weight)
        self.edit_record.append(record)

    # Function to create new material text file
    def create_material(self, location, filename, material_info):
        # check to see if file already exists

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
            d = str((wave[i] / 1000)), ' ', str(values_single[i]), '\n'
            g.writelines(d)
        g.close()

    # Function to export image to a file
    def export(self, image_name = 'Untitled'):
        export_im = deepcopy(self)
        save_to = 'Anomaly Files/'
        data = export_im.data.astype(">i2")
        f = open(save_to + image_name, "wb")
        f.write(data)

        self.write_HDR(save_to, image_name)
        # self.write_record_file(save_to, image_name)

    # -------------------------------------------------------------------------------
    # EXPLORE FUNCTIONS
    # -------------------------------------------------------------------------------
    # Function to create a material from single pixel in image
    def graph_spectra_pixel(self, location, title, single):
        values_single = []
        for i in range(self.img_bands):
            values_single.append(self.data[location[1], location[0], i])

        # print(self.wavelengths_dict.values())

        try:
            plt.plot(list(self.wavelengths_dict.values()), values_single, linewidth=2, label=title)

        except:
            x_a = np.linspace(0, self.img_bands, num=self.img_bands)
            plt.plot(x_a, values_single, linewidth=2, label=title)
        plt.xlabel('Bands')
        plt.ylabel('Values')
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

    def mono_pika_comp(self):

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
        diff = abs(pika_band - mono_band)
        if print: print(f'Mono Band: {mono_band} / Pika: {pika_band} / Diff = {diff}')

        return diff



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
    def display_RGB(self, display):
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
        if display == True:
            # plt.figure(figsize=(18, 8))
            plt.imshow(rgb_stack, cmap=plt.get_cmap(None))
            plt.axis('off')
            plt.show()
        else:
            self.rgb_graph = rgb_stack

    # Function to display the NDVI
    def display_NDVI(self, display):
        # print(self.wavelengths_dict)
        # find bands nearest to NDVI red and nir wavelengths
        # OG bands used 36 (685) / 58 (900)
        # Trying 30-650 / 45-770
        if self.file_name.lower() == 'botswana':
            ndvi_nir = 28
            ndvi_red = 20
        else:
            ndvi_nir = 52
            ndvi_red = 32

        RED = self.data[:, :, ndvi_red]
        NIR = self.data[:, :, ndvi_nir]
        RED, NIR = RED.astype('float64'), NIR.astype('float64')
        RED[RED == -50.0], NIR[NIR == -50.0] = np.nan, np.nan

        # calculate ndvi
        ndvi_array = (NIR - RED) / (NIR + RED)

        # plot. all of this is matplotlib ---------->
        ndvi_array_temp = np.zeros(ndvi_array.shape, dtype=float)
        ndvi_array_temp[ndvi_array >= 0.1] = ndvi_array[ndvi_array >= 0.1]

        if display:
            # plt.figure(figsize=(10, 15))
            plt.imshow(ndvi_array_temp, vmax=1, cmap=plt.get_cmap("RdYlGn"))
            plt.title('NDVI')
            plt.colorbar()
            plt.axis('off')
            plt.show()
        else:
            self.graph_ndvi = ndvi_array_temp

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

        # plt.figure(figsize=(18, 8))
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

        # plt.figure(figsize=(18, 8))
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
        plt.imshow(ndwi_array_temp, vmax=1, cmap=plt.get_cmap("Blues"))
        plt.title('NDWI')
        plt.colorbar()
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
    def display_subcategories(self, display, cutoff_percent=100):

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

    # Function to create a material from single pixel in image
    def subcategorize_pixels_1(self, band='M', variation=20, iterations=15, av=False):
        total_start_time = time.time()
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
        print('Initializing Pixel Objects')
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

        print(f'Total Time: {(time.time() - total_start_time) / 60} mins')

    # Function to be able to perform multiple operations on images at with one process
    def analysis(self):
        run = True
        self.subcategorize_pixels_1()

        while run:
            command = input('Enter Command: ')
            command.lower()

            try:
                if command == 'rgb':
                    self.display_RGB()
                elif command == 'ndvi':
                    self.display_NDVI()
                elif command == 'ndwi':
                    self.display_NDWI()
                elif command == 'index':
                    self.display_indices()
                elif command == 'graph':
                    self.graph_spectra_all_pixels()
                elif command == 'subcat':
                    self.display_subcategories()
                elif command == 'gasc':
                    self.graph_average_subcat()
                elif command == 'compare':
                    self.compare()
                elif command == 'help':
                    print('rgb: Display RGB Image')
                    print('ndvi: Display Diff Norm Veg Index')
                    print('ndwi: Display Diff Norm Water Index')
                    print('index: Display Indicies')
                    print('graph: Graph all pixels in image')
                    print('subcat: Display the Subcategories')
                    print('gasc: Graph Average values of each Subcategory')
                elif command == 'x':
                    run = False
                else: print('Input Not Recognized')

            except:
                print('Error Running Function - Try Another')

    # Function to be able to perform multiple operations on images at with one process
    def compare(self,display, save):
        self.subcategorize_pixels_1()
        self.display_RGB(False)
        self.display_indices(False)
        self.display_subcategories(display=False, cutoff_percent=100)
        self.display_NDVI(False)

        plt.figure(figsize=(15, 8))
        plt.suptitle(f'{self.file_name} Comparison')

        plt.subplot(2, 3, 1)
        plt.imshow(self.rgb_graph, cmap=plt.get_cmap(None))
        plt.title('RGB')
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.imshow(self.graph_ndvi, cmap=plt.get_cmap("RdYlGn"))
        plt.title('NDVI')
        # plt.colorbar()
        plt.axis('off')

        plt.subplot(2, 3, 3)
        og = plt.imshow(self.graph_indicies[0], cmap='gray')
        veg = plt.imshow(self.graph_indicies[1], cmap='Greens')
        wat = plt.imshow(self.graph_indicies[2], cmap='Blues')
        c1 = plt.colorbar(og, cmap='gray')
        c2 = plt.colorbar(veg, cmap='Greens')
        c3 = plt.colorbar(wat, cmap='Blues')
        c1.remove()
        c2.remove()
        c3.remove()
        plt.title('Indicies')
        plt.axis('off')

        plt.subplot(2, 3, 4)
        plt.imshow(self.graph_subcat, cmap='nipy_spectral')
        plt.title('Subcategories')
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.title('All Pixels')
        x_list = np.linspace(0, (self.img_x - 1), 100)
        y_list = np.linspace(0, (self.img_y - 1), 100)

        for i in x_list:
            for j in y_list:
                self.graph_spectra_pixel([int(i), int(j)], 'Full', False)

        plt.subplot(2, 3, 6)
        plt.title('Average Subcategories')
        for subcat in self.subcat_master_list:
            x_a = np.linspace(0, self.img_bands, num=self.img_bands)
            plt.plot(x_a, subcat.average_values, linewidth=2)
            # plt.xlabel('Bands')
            # plt.ylabel('Values')

        if save:
            saveas = (f'../../Dropbox/2 Work/1 Optics Lab/1 Anomaly Detection/_Cassification/Autosaves/{self.file_name}-Compare')
            plt.savefig(saveas)
            plt.close()
        if display:
            plt.show()

    # -------------------------------------------------------------------------------
    # ANOMALY DETECTOR FUNCTIONS
    # -------------------------------------------------------------------------------

    def RX_Spectral(self):

        try:
            spectral.settings.envi_support_nonlowercase_params = True
            # self.img = spectral.io.envi.open(hdr_file_path)
            self.img = spectral.open_image(self.file_path + '.hdr')
            self.data = self.img.load()
            self.data = np.where(self.data < 0, 0, self.data)

            # Water Vapour Absorption Band, Water Vapour Absorption Band, Not Illuminated
            if 105 < self.img.nbands < 225:
                w_bands = list(range(105, 120)) + list(range(150, 171)) + list(range(215, 224))
                self.data = np.delete(self.data, w_bands, axis=2)

            inc = 500

            rxvals = None
            for i in range(0, self.data.shape[0], inc):
                rxrow = None

                for j in range(0, self.data.shape[1], inc):
                    d = self.data[i:i + inc, j:j + inc, :]
                    rxcol = spectral.rx(d)  # spectral.rx(data, window=(5,211))  Local RX
                    rxrow = np.append(rxrow, rxcol, axis=1) if rxrow is not None else rxcol
                rxvals = np.vstack((rxvals, rxrow)) if rxvals is not None else rxrow

            nbands = self.data.shape[-1]
            P = chi2.ppf(0.999, nbands)  # 0.998

            v = spectral.imshow(self.img, bands=(30, 20, 10), figsize=(12, 6), classes=(1 * (rxvals > P)))
            v.set_display_mode('overlay')
            v.class_alpha = 0.7  # how transparent the overlaid portion is
            plt.title('RX-Seg Algorithm')
            plt.axis('off')
            plt.pause(500)

        except:
            print('Not an ENVI File')

    def Anomaly_Custom(self):
        from sklearn.ensemble import IsolationForest
        import matplotlib.colors

        self.display_RGB(False)
        # Reshape the data array
        data = self.data.reshape(-1, self.data.shape[-1])

        # Create an instance of the RX anomaly detector and fit it to the data
        detector = IsolationForest(contamination=0.1)
        #default is 0.1: 10% will be anomalous
        detector.fit(data)

        # Use the detector to predict anomalies in the data
        labels = detector.predict(data)

        # Create a new image containing only the anomalous points
        anomalous_image = data[labels == -1]

        # Create a new array with the same shape as the original data array
        result = np.zeros_like(data)

        # Copy the anomalous points from the anomalous_image array into the result array
        result[labels == -1] = anomalous_image

        # Take the mean of all bands for each pixel
        result = result.mean(axis=-1)

        # Reshape the result array to the original shape of the data
        result = result.reshape(self.data.shape[0], self.data.shape[1])

        # Create a color map with red for anomalous pixels and white for normal pixels
        cmap = matplotlib.colors.ListedColormap(['white', 'red'])

        # Display the result array using the color map
        plt.imshow(self.rgb_graph, cmap=plt.get_cmap(None))
        plt.title('RGB')
        plt.axis('off')
        plt.imshow(result, cmap=cmap)
        plt.colorbar()
        plt.show()


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



