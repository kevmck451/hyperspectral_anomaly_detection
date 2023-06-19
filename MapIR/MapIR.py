#Analysis of Data from MapIR Camera
#Kevin McKenzie 2022

import matplotlib.pyplot as plt
from copy import deepcopy
from PIL import Image
import numpy as np
import tifffile
import cv2
import os
from pathlib import Path
import piexif
import imageio


save_to_location = '../2 MapIR/Autosaves'

# RAW DATA CONVERSION CLASS
class MapIR_RAW:
    def __init__(self, raw_file_path, stats=False):
        self.debay = False
        self.unp = False
        self.file_path = raw_file_path
        self.corrected = False

        path = Path(raw_file_path)
        self.file_name = path.stem
        self.file_type = path.suffix

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
            # n = self.file_path
            # n = n.split('/')
            # try:
            #     n = n[-1]
            #     n = n.split('.')
            # except:
            #     pass

            # self.band = n[0]
            # print(self.band)

            self._debayer()
            self._correct()

            # self.white_balance()
            # self.render_RGB()

            # Load the image as a numpy array
            data = self.data

            # normalize to 0-1 range
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))

            # now scale to 0-65535 range (which is the range of uint16)
            data_scaled = data_norm * 65535

            # finally, convert to uint16
            self.data = data_scaled.astype(np.uint16)

            # print(image_array)
            # print(image_array.dtype)

            if stats:
                print(f'Y: {self.img_y} / X: {self.img_x} / Bands: {self.img_bands}')


        except:
            print(f'File Corrupt: {self.file_name}')

    # Function to Debayer image matrix
    def _debayer(self):

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
    def _white_balance(self):
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

    # Function to calibrate MAPIR Camera for analysis
    def _correct(self):
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
    def render_RGB(self, hist=False):
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

    # Function to display the data
    def display(self, hist=False):

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
        plt.tight_layout(pad=1)
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
    def export(self):

        directory = '/'.join(self.file_path.split('/')[:-1])
        directory = f'{directory}/Corr'
        save_name = f'{self.file_name}_C.png'
        self.render_RGB()
        data = self.rgb_render

        # Ensure directory exists, if not create it
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Define the path to save the image
        path = os.path.join(directory, save_name)

        # Write the image data to a file
        image = Image.fromarray(data)
        image.save(path)

    # Function to export processed data
    def export_all(self):

        directory = '/'.join(self.file_path.split('/')[:-1])
        directory_ndvi = f'{directory}/NDVI'
        directory = f'{directory}/Corr'
        save_name = f'{self.file_name}_C.png'
        save_name_ndvi = f'{self.file_name}_NDVI.png'
        self.render_RGB()
        data = self.rgb_render
        self.NDVI(display=False, save=False)
        data_ndvi = self.ndvi

        # Ensure directory exists, if not create it
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Define the path to save the image
        path = os.path.join(directory, save_name)

        # Write the image data to a file
        image = Image.fromarray(data)
        image.save(path)

        # Ensure directory exists, if not create it
        if not os.path.exists(directory_ndvi):
            os.makedirs(directory_ndvi)

        # Define the path to save the image
        path = os.path.join(directory_ndvi, save_name_ndvi)

        # Write the image data to a file
        plt.imshow(data_ndvi, cmap=plt.get_cmap("RdYlGn"))
        plt.title(f'NDVI')
        plt.axis('off')
        plt.tight_layout(pad=1)
        plt.savefig(path)
        plt.close()

    # Function to display the Normalized Difference Veg Index
    def NDVI(self, display=True, save=False):

        NIR = self.data[:, :, self.NIR_index]
        RED = self.data[:, :, self.R_index]

        RED, NIR = RED.astype('float'), NIR.astype('float')
        # RED[RED == 0], NIR[NIR == 0] = np.nan, np.nan
        top, bottom = NIR - RED, NIR + RED
        top[top == 0], bottom[bottom == 0] = 0, np.nan

        ndvi_array = np.divide(top, bottom)
        ndvi_array[ndvi_array < 0] = 0
        ndvi_array[ndvi_array > 1] = 1
        self.ndvi = ndvi_array


        plt.imshow(ndvi_array, cmap=plt.get_cmap("RdYlGn"))
        plt.title(f'NDVI: {self.file_name}')
        # plt.colorbar()
        plt.axis('off')
        plt.tight_layout(pad=1)

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

    # Function to extract GPS metadata from corresponding jpg image
    def extract_GPS(self, file_type):

        path = Path(self.file_path)
        jpg_num = int(path.stem) + 1
        if len(str(jpg_num)) != 3:
            jpg_num = '0' + str(jpg_num)
        jpg_filepath = f'{path.parent}/{jpg_num}.jpg'
        image = Image.open(jpg_filepath)

        exif_data = piexif.load(image.info["exif"])

        # Extract the GPS data from the GPS portion of the metadata
        geolocation = exif_data["GPS"]

        # Create a new NumPy array with float32 data type and copy the geolocation data
        geolocation_array = np.zeros((3,), dtype=np.float32)
        if geolocation:
            latitude = geolocation[piexif.GPSIFD.GPSLatitude]
            longitude = geolocation[piexif.GPSIFD.GPSLongitude]
            altitude = geolocation.get(piexif.GPSIFD.GPSAltitude, [0, 1])  # Add this line

            if latitude and longitude and altitude:  # Add altitude check here
                lat_degrees = latitude[0][0] / latitude[0][1]
                lat_minutes = latitude[1][0] / latitude[1][1]
                lat_seconds = latitude[2][0] / latitude[2][1]

                lon_degrees = longitude[0][0] / longitude[0][1]
                lon_minutes = longitude[1][0] / longitude[1][1]
                lon_seconds = longitude[2][0] / longitude[2][1]

                altitude_val = altitude[0] / altitude[1]  # altitude calculation

                geolocation_array[0] = lat_degrees + (lat_minutes / 60) + (lat_seconds / 3600)
                geolocation_array[1] = lon_degrees + (lon_minutes / 60) + (
                            lon_seconds / 3600)  # updated with lon minutes and seconds
                geolocation_array[2] = altitude_val  # assign altitude to array


            # Append the image geolocation to the geo.txt file
            file_path = os.path.join(path.parent, '_processed', 'geo.txt')

            # Check if the file exists
            if not os.path.exists(file_path):
                # If the file doesn't exist, it is the first time it is being opened
                # Write the header "EPSG:4326"
                with open(file_path, 'w') as f:
                    f.write('EPSG:4326\n')

            # Append the data to the file
            with open(file_path, 'a') as f:
                f.write(f'{path.stem}.{file_type}\t-{geolocation_array[1]}\t{geolocation_array[0]}\t{geolocation_array[2]}\n')

    # Function to export image as tiff
    def export_tiff(self):
        path = Path(self.file_path)
        save_as = f'{path.parent}/_processed/{path.stem}.tiff'
        imageio.imsave(save_as, self.data, format='tiff')

    # Function to export image as 16-bit png
    def export_png(self):
        path = Path(self.file_path)
        save_as = path.parent/"_processed"/(path.stem+".png")
        imageio.imwrite(save_as, self.data, 'PNG-FI')





















