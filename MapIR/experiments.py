

import matplotlib.pyplot as plt
from copy import deepcopy
import imageio.v3 as iio
from PIL import Image
import numpy as np
import time
import cv2
import os
from MapIR import MapIR_RAW
from jpg import MapIR_jpg


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