#Hyperspectral is used to create HSI objects and functions

from Materials.Materials import Material as m
import matplotlib.gridspec as gridspec
from sklearn import preprocessing
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import numpy as np
import shutil

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
            wave = float(line[0].strip())
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
        plt.title('IF-Band: {} nm'.format(self.wavelengths_dict.get(band_list[0])), fontsize=font_size)
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

        # function to classify bands
        between = lambda wavelength, region: region['lower'] < wavelength <= region['upper']

        def classifier(band):
            for region, limits in Hyperspectral.wavelength_color_dict.items():
                if between(band, limits):
                    return (region)

        # lists of band numbers, band centers, and em classes
        band_numbers = [i for i in range(1, len(self.wavelengths) + 1)]
        em_regions = [classifier(b) for b in self.wavelengths]

        # data frame describing bands
        bands = pd.DataFrame({
            "Band number": band_numbers,
            "Band center": self.wavelengths,
            "EM region": em_regions}, index=band_numbers).sort_index()

        # function finds band in our table with wavelength nearest to input r,g,b wavelengths
        get_band_number = lambda w: bands.iloc[(bands["Band center"] - w).abs().argsort()[1]]

        scale8bit = lambda a: ((a - a.min()) * (1 / (a.max() - a.min()) * 255)).astype('uint8')
        # print(self.wavelengths_dict)
        # get band numbers. use R: 600, G: 500, B: 400

        # Ri, Gi, Bi = get_band_number(667.5), get_band_number(540), get_band_number(470)
        Ri = 25  # 35 #25 #29 #32
        Gi = 20  # 20 #15 #17 #19
        Bi = 15  # 17 #5 #10 #12

        # get r,g,b arrays
        Ra = self.data[:, :, Ri]
        Ga = self.data[:, :, Gi]
        Ba = self.data[:, :, Bi]

        # set fill values (-9999.) to 0 for each array
        Ra[Ra == -50], Ga[Ga == -50], Ba[Ba == -50] = 0, 0, 0

        # get 8bit arrays for each band
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

        # %matplotlib inline
        titlefont = {'fontsize': 16, 'fontweight': 2,
                     'verticalalignment': 'baseline', 'horizontalalignment': 'center'}

        # plot. all of this is matplotlib ---------->
        plt.rcParams['figure.figsize'] = [12, 8]
        gs = gridspec.GridSpec(1, 4)

        plotdict = {'Red': {'subplot': 0, 'array': rgb_stack[:, :, 0], 'colormap': 'Reds_r'},
                    'Green': {'subplot': 1, 'array': rgb_stack[:, :, 1], 'colormap': 'Greens_r'},
                    'Blue': {'subplot': 2, 'array': rgb_stack[:, :, 2], 'colormap': 'Blues_r'},
                    'RGB': {'subplot': 3, 'array': rgb_stack, 'colormap': None}}

        # initialize plot and add ax element for each array in plotdict
        fig1 = plt.figure()
        for band, data in plotdict.items():
            clim = None if band == "RGB" else (0, 255)
            ax = fig1.add_subplot(gs[0, data['subplot']])
            p = ax.imshow(data['array'], cmap=data['colormap'], clim=clim)
            ax.set_title(band, pad=20, fontdict=titlefont)
            plt.axis('off')
        plt.imshow(rgb_stack)
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

        R750_b = 43
        R705_b = 38
        R750 = self.data[:, :, R750_b]
        R705 = self.data[:, :, R705_b]

        # calculate sr
        sr_array = R750 / R705

        L = 0.5
        # calculate savi
        savi_array = (1 + L) * ((NIR - RED) / (NIR + RED + L));

        titlefont = {'fontsize': 16, 'fontweight': 2,
                     'verticalalignment': 'baseline', 'horizontalalignment': 'center'}

        plt.rcParams['figure.figsize'] = [12, 8]

        # plot. all of this is matplotlib ---------->
        gs = gridspec.GridSpec(1, 3)

        ndvi_array_temp = np.zeros(ndvi_array.shape, dtype=float)
        ndvi_array_temp[ndvi_array >= 0.1] = ndvi_array[ndvi_array >= 0.1]

        plotdict1 = {'NDVI (threshold)': {'subplot': 0, 'array': ndvi_array_temp},
                     'NDVI': {'subplot': 1, 'array': ndvi_array},
                     'SAVI': {'subplot': 2, 'array': savi_array}}

        # initialize plot and add ax element for each array in plotdict
        fig2 = plt.figure()
        for b, data in plotdict1.items():
            ax = fig2.add_subplot(gs[0, data['subplot']])

            p = ax.imshow(data['array'], cmap=plt.get_cmap("RdYlGn"))
            ax.set_title(b, pad=5, fontdict=titlefont)
            plt.colorbar(p)
            plt.axis('off')

        fig2.subplots_adjust(wspace=0, hspace=0)
        plt.show()

    # Function to add anomaly using material & image as is
    def add_anomaly_1(self, material, location, size, scale_factor):
        mat = material.map_material_to_image(list(self.wavelengths_dict.values()))
        x_list = [(location[0] - size), (location[0] + size)]
        y_list = [location[1] + size, location[1] - size]

        for i in range(0, len(mat)):
            self.data[y_list[1]:y_list[0], x_list[0]:x_list[1], i] = (mat[i]*scale_factor)
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

        for i in range(0, len(mat)):
            self.data[y_list[1]:y_list[0], x_list[0]:x_list[1], i] = adjusted[i]
        self.anom_record.append('A3_{}'.format(material.material_id))

    # Function to export image to a file
    def export(self, image_name):
        export_im = deepcopy(self)
        save_to = 'Anomaly Files/'
        data = export_im.data.astype(">i2")
        f = open(save_to + image_name, "wb")
        f.write(data)
        g = open(save_to + image_name + '.hdr', 'w')
        g.writelines('ENVI\n')
        for x, y in self.header_file_dict.items():
            d = x, ' = ', y, '\n'
            g.writelines(d)
        g.writelines('wavelength = {\n')
        for x in self.wavelengths:
            d = str(x), ' ,\n'
            g.writelines(d)

        # shutil.copyfile(self.hdr_file_path, (save_to + image_name + '.hdr'))

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

        for i in range(224):
            for j in range(400, 2600):
                for k in range(200, 600):
                    area_values.append(self.data[j, k, i])

            nums = np.array(area_values)
            min.append(nums.min())
            max.append(nums.max())
            av.append(nums.mean())
            area_values.clear()

        min = np.array(min)
        max = np.array(max)
        av = np.array(av)
        min_total = min.min()
        max_total = max.max()
        av_total = av.mean()
        min_max_av_list = [min, max, av]

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











