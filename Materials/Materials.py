#Materials is used to create a Material Library and Material Objects
#Kevin McKenzie 2022

import matplotlib.pyplot as plt
import os

class Material_Lib:
    type_list = ['manmade','meteorites','mineral','non photosynthetic vegetation','vegetation','rock','soil','water','custom']
    material_type_abrev = ['MM', 'ME', 'MI', 'NV', 'VG', 'RK', 'SL', 'WR', 'CM']

    def __init__(self):

        directory = 'Materials/Material Data'
        self.material_list_full = []
        self.manmade_list = []
        self.meteorites_list = []
        self.mineral_list = []
        self.non_photosynthetic_vegetation_list = []
        self.vegetation_list = []
        self.rock_list = []
        self.soil_list = []
        self.water_list = []
        self.custom_list = []
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            if os.path.isfile(f):
                try:
                    self.material_list_full.append(Material(filename[:-4]))
                except:
                    continue

        self.create_types()

    #Removes all bands outside of argument:
    #Argument list with x_y[0] being low end and x_y[1] being high end (in nanometers)
    def filter(self,x_y):
        low_pass_dict = {}
        remove_list = []
        for x in self.material_list_full:
            for i in range(x_y[0],(x_y[1]+1)):
                if x.banddict.get(i):
                    low_pass_dict.update({i : x.banddict.get(i)})
                else:
                    continue
            if len(low_pass_dict) == 0:
                remove_list.append(x)
            x.banddict = low_pass_dict
            low_pass_dict = {}

        for x in remove_list:
            self.material_list_full.remove(x)

        self.create_types()

    #Creates lists of materials by type
    def create_types(self):
        for y in self.material_list_full:
            z = y.infodict.get('Type')
            z = z.lower()
            if z == Material_Lib.type_list[0]:
                self.manmade_list.append(y)
            elif z == Material_Lib.type_list[1]:
                self.meteorites_list.append(y)
            elif z == Material_Lib.type_list[2]:
                self.mineral_list.append(y)
            elif z == Material_Lib.type_list[3]:
                self.non_photosynthetic_vegetation_list.append(y)
            elif z == Material_Lib.type_list[4]:
                self.vegetation_list.append(y)
            elif z == Material_Lib.type_list[5]:
                self.rock_list.append(y)
            elif z == Material_Lib.type_list[6]:
                self.soil_list.append(y)
            elif z == Material_Lib.type_list[7]:
                self.water_list.append(y)
            elif z == Material_Lib.type_list[8]:
                self.custom_list.append(y)
            else:
                continue

class Material:
    infonames = ['Name', 'Type', 'Class', 'Subclass', 'Particle Size', 'Sample No.', 'Owner',
                 'Wavelength', 'Origin', 'Collection Date', 'Description', 'Measurement',
                 'First Column', 'Second Column', 'X Units', 'Y Units', 'First X Value',
                 'Last X Value', 'Number of X Values', 'Additional Info']
    directory = 'Materials/Material Data/'

    def __init__(self, material_id):
        self.material_id = material_id
        materialfile = open((Material.directory + material_id + '.txt'), 'r', errors='ignore')
        self.infolist = []
        self.infodict = {}
        for i in range(1, 21):  # 1, 21
            k = materialfile.readline()
            k = k.split(':')
            k = k[1]
            k = k.strip()
            self.infolist.append(k)
            self.infodict.update({Material.infonames[i - 1]: self.infolist[i - 1]})

        self.banddict = {}
        materialfile.readline()
        for i in range(22, int(self.infolist[18]) + 22):
            k = materialfile.readline()
            k = k.split()
            k[0] = float(k[0])
            k[0] = k[0] * 1000
            self.banddict.update({k[0] : float(k[1])})

    # Graph the Raw Material Spectral Profile
    def graph_material(self, single):
        # plt.figure(figsize = (18,6))
        plt.plot(list(self.banddict.keys()), list(self.banddict.values()), linewidth=2, label=self.infodict['Name'])
        plt.xlabel('Bands')
        plt.ylabel('Values')
        plt.title(self.infodict['Type'] + ': Full Profile', fontsize=20)
        plt.legend(loc='upper right')
        if single:
            plt.show()

    #Maps the material's profile to the number of image data bands
    def map_material_to_image(self, image_wavelength_list):
        mapped_data = []
        new_dict = {}
        for wave in image_wavelength_list:
            a = wave
            found = False
            while not found:
                for w in self.banddict:
                    if w == a:
                        mapped_data.append(self.banddict.get(w))
                        found = True
                        break
                a = a + 1
        return mapped_data

    #Graph the Raw Material Spectral Profile and Altered Profile that fits Data
    def graph_mapped_materials(self, image_wavelength_list):
        plt.figure(figsize=(18, 6))
        plt.subplot(121)
        plt.plot(list(self.banddict.keys()), list(self.banddict.values()), linewidth=3)
        plt.axvline(x=366, color='r', linestyle='--', linewidth=1)
        plt.axvline(x=2496, color='r', linestyle='--', linewidth=1)
        plt.xlabel('Bands')
        plt.ylabel('Values')
        plt.title(self.infodict['Name'] + ': Full Profile', fontsize=20)
        plt.subplot(122)
        plt.plot(image_wavelength_list, self.mapped_data, linewidth=3)
        plt.xlabel('Bands')
        plt.ylabel('Values')
        plt.title(self.infodict['Name'] + ': Section of Profile', fontsize=20)
        plt.show()


