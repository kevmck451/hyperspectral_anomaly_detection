from Materials.Materials import Material as m
from Materials.Materials import Material_Lib
from Hyperspectral import Hyperspectral as h
from Recon import Recon as rc

'''
#Creating a material library object, low pass function, graphing all manmade materials
mat_lib = Material_Lib()

#filer
x_y = [400,1000]
mat_lib.filter(x_y)

for x in mat_lib.manmade_list:
    x.graph_material(False)
plt.show()

'''

'''
A_file_path = 'AVIRIS/f100704t01p00r06rdn_b_sc01_ort_img'
image = h(A_file_path) #a hyperspectral image object using hyperspectral
image_og = h(A_file_path)

material_brick = m('1MM0008') #a material object using material
material_steel = m('1MM0017')
material_concrete = m('1MM0029')
material_asphalt = m('1MM0006')
material_tar = m('1MM0027')
material_lumber_paint_green = m('1MM0015')
material_lumber_paint_black = m('1MM0004')
material_black_tar_paper = m('1MM0012')
location_1 = [437, 1779] #dictionary of key-value pair of x-y coordinate of middle of anomaly
location_2 = [470, 1950]
location_3 = [550, 2350]
size = 30   #the distance from the center the shape will extend

image.add_anomaly_2(material_lumber_paint_green, location_1, size)
image.add_anomaly_2(material_brick, location_2, size+10)
image.add_anomaly_2(material_concrete, location_3, size+30)

#tank
image.add_anomaly_2(material_black_tar_paper, [550, 2350], 20)
image.add_anomaly_2(material_black_tar_paper, [560, 2350], 20)
image.add_anomaly_2(material_black_tar_paper, [525, 2350], 5)
image.add_anomaly_2(material_black_tar_paper, [515, 2350], 5)
image.add_anomaly_2(material_black_tar_paper, [505, 2350], 5)

image_og.display_image()
image.display_image()
image.display_RGB()
image.display_Veg_Index()

# image.export()
'''

'''
anom_test_1 = 'Anomaly Files/AVIRIS_A1_Test1.hdr'

recon_1 = rc(anom_test_1)
recon_1.rx_seg()
'''

con = m('1MM0029')
print(con.banddict)