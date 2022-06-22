#THIS PROGRAM DELETES ANY FILE THAT CANT MAKE AN OBJECT
#RENAMES THE FILES WITH THEIR MATERIAL ID
#CREATES A MATERIAL MENU FILE USED TO CREATE INDIVIDUAL OBJECTS IN MATERIAL CLASS

'''
import os
import json
from Materials import Material as m

material_list_full = []
type_list = ['manmade','meteorites','mineral','non photosynthetic vegetation','vegetation','rock','soil','water']
material_type_abrev = ['1MM', '8ME', '7MI', '5NV', '6VG', '4RK', '3SL', '2WR']
manmade_list = []
meteorites_list = []
mineral_list = []
non_photosynthetic_vegetation_list = []
vegetation_list = []
rock_list = []
soil_list = []
water_list = []
directory = 'Material Data'

starting_id_0 = 1
starting_id_1 = 1
starting_id_2 = 1
starting_id_3 = 1
starting_id_4 = 1
starting_id_5 = 1
starting_id_6 = 1
starting_id_7 = 1

Material_Menu = {}

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        try:
            o = m(f)
            material_list_full.append(o)
            z = o.infodict.get('Type')
            z = z.lower()
            if z == type_list[0]:
                # name & sub class for mm
                a = material_type_abrev[0]
                material_id = '{}{}'.format(a, '{:04d}'.format(starting_id_0))
                starting_id_0 += 1
                Material_Menu.update({material_id : o.infodict.get('Name')})
                #rename file: file name is material id
                path = 'Material Data/' + material_id + '.txt'
                os.renames(f,path)
            elif z == type_list[1]:
                # name & sub class for mm
                a = material_type_abrev[1]
                material_id = '{}{}'.format(a, '{:04d}'.format(starting_id_1))
                starting_id_1 += 1
                Material_Menu.update({material_id : o.infodict.get('Name')})
                # rename file: file name is material id
                path = 'Material Data/' + material_id + '.txt'
                os.renames(f, path)
            elif z == type_list[2]:
                # name & sub class for mm
                a = material_type_abrev[2]
                material_id = '{}{}'.format(a, '{:04d}'.format(starting_id_2))
                starting_id_2 += 1
                Material_Menu.update({material_id : o.infodict.get('Name')})
                # rename file: file name is material id
                path = 'Material Data/' + material_id + '.txt'
                os.renames(f, path)
            elif z == type_list[3]:
                # name & sub class for mm
                a = material_type_abrev[3]
                material_id = '{}{}'.format(a, '{:04d}'.format(starting_id_3))
                starting_id_3 += 1
                Material_Menu.update({material_id : o.infodict.get('Name')})
                # rename file: file name is material id
                path = 'Material Data/' + material_id + '.txt'
                os.renames(f, path)
            elif z == type_list[4]:
                # name & sub class for mm
                a = material_type_abrev[4]
                material_id = '{}{}'.format(a, '{:04d}'.format(starting_id_4))
                starting_id_4 += 1
                Material_Menu.update({material_id : o.infodict.get('Name')})
                # rename file: file name is material id
                path = 'Material Data/' + material_id + '.txt'
                os.renames(f, path)
            elif z == type_list[5]:
                # name & sub class for mm
                a = material_type_abrev[5]
                material_id = '{}{}'.format(a, '{:04d}'.format(starting_id_5))
                starting_id_5 += 1
                Material_Menu.update({material_id : o.infodict.get('Name')})
                # rename file: file name is material id
                path = 'Material Data/' + material_id + '.txt'
                os.renames(f, path)
            elif z == type_list[6]:
                # name & sub class for mm
                a = material_type_abrev[6]
                material_id = '{}{}'.format(a, '{:04d}'.format(starting_id_6))
                starting_id_6 += 1
                Material_Menu.update({material_id : o.infodict.get('Name')})
                # rename file: file name is material id
                path = 'Material Data/' + material_id + '.txt'
                os.renames(f, path)
            elif z == type_list[7]:
                # name & sub class for mm
                a = material_type_abrev[7]
                material_id = '{}{}'.format(a, '{:04d}'.format(starting_id_7))
                starting_id_7 += 1
                Material_Menu.update({material_id : o.infodict.get('Name')})
                # rename file: file name is material id
                path = 'Material Data/' + material_id + '.txt'
                os.renames(f, path)
            else:
                continue
        except:
            #REMOVE FILE IF YOU CANT MAKE AN OBJECT FROM IT
            continue

Material_Menu_Sorted = dict(sorted(Material_Menu.items(), key=lambda item: item[0]))

with open('Material_Menu.txt', 'w') as f:
    for key, value in Material_Menu_Sorted.items():
        f.write('%s : %s\n' % (key, value))

#print(Material_Menu)
'''


