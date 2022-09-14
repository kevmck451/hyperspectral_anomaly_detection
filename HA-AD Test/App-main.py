# Human Assisted Anomaly Detection Test File

import pygame
from pygame.locals import *
import sys
from stuff import button
from stuff import image as i
from stuff import text

# call back functions
def rgb_display():
    RGB_image = i(image=pygame.image.load('Images/Test Images/RGB-3A.png'), position=(width / 2, height / 2), scale=0.5)
    RGB_image.draw(screen)
    rgb_text = button((width / 2, height/8), (80, 25), pygame.Color('gray'), pygame.Color('gray'),'R-G-B', "Arial", 25, [0, 0, 0])
    rgb_text.draw(screen)
def ndvi_display():
    NDVI_image = i(image=pygame.image.load('Images/Test Images/VI-3A.png'), position=(width / 2, height / 2), scale=0.5)
    NDVI_image.draw(screen)
    ndvi_text = button((width / 2, height/8), (80, 25), pygame.Color('gray'), pygame.Color('gray'),'N-D-V-I', "Arial", 25, [0, 0, 0])
    ndvi_text.draw(screen)
def rxd_display():
    RXD_image = i(image=pygame.image.load('Images/Test Images/RX-3A.png'), position=(width / 2, height / 2), scale=0.5)
    RXD_image.draw(screen)
    rxd_text = button((width / 2, height/8), (80, 25), pygame.Color('gray'), pygame.Color('gray'),'R-X-D', "Arial", 25, [0, 0, 0])
    rxd_text.draw(screen)
def plus_parameter():
    print('PLUS')
def minus_parameter():
    print('MINUS')
def mouse_position():
    pos = pygame.mouse.get_pos()
    pos = str(pos)
    mouse_pos = button((width / 2, height * 31 / 32), (100, 20), pygame.Color('gray'), pygame.Color('gray'), pos, "Arial", 20, [0, 0, 0])
    mouse_pos.draw(screen)








if __name__ == '__main__':

    # ---------------------------------------------------------------------------------------------
    '''APP DETAILS'''
    # ---------------------------------------------------------------------------------------------

    pygame.init()
    pygame.display.set_icon(pygame.image.load('Images/Recon Icon.png'))
    pygame.display.set_caption('Human Assisted Anomaly Detection')
    screen_size = (1200, 800) #1380, 840
    font_size   = 15
    font        = pygame.font.Font(None, font_size)
    clock       = pygame.time.Clock()
    screen    = pygame.display.set_mode(screen_size)
    width = screen.get_width()
    height = screen.get_height()

    bg = i(image=pygame.image.load('Images/Background.png'), position=(0, 0), scale=1, mid = False)
    logo = i(image=pygame.image.load('Images/Recon.png'), position=(0, 0), scale=0.5, mid = False)


    # TESTING PAGE -----------------------------------------
    screen.fill(pygame.Color('white'))
    bg.draw(screen)
    logo.draw(screen)
    RGB_button = button(position=( width*7 / 8 , height*3 / 32), size=(200, 60), clr=pygame.Color('gray'), cngclr=pygame.Color('dark gray'), text='R-G-B', font="Arial", font_size=45, font_clr=[0, 0, 0])
    NDVI_button = button((width*7 / 8  , height*6 / 32), (200, 60), pygame.Color('gray'), pygame.Color('dark gray'), 'N-D-V-I', "Arial", 45, [0, 0, 0])
    RXD_button = button((width*7 / 8  , height*9 / 32), (200, 60), pygame.Color('gray'), pygame.Color('dark gray'), 'R-X-D', "Arial", 45, [0, 0, 0])
    plus_button = button((width * 30 / 32, height * 11 / 32), (30, 30), pygame.Color('gray'), pygame.Color('dark gray'), '+', "Arial", 45, [0, 0, 0])
    minus_button = button((width *26 / 32, height * 11 / 32), (30, 30), pygame.Color('gray'), pygame.Color('dark gray'), '-', "Arial", 45, [0, 0, 0])

    button_list = [RGB_button, NDVI_button, RXD_button, plus_button, minus_button]



    #---------------------------------------------------------------------------------------------
    '''APP LOOP'''
    #---------------------------------------------------------------------------------------------

    crash = True
    first = True
    while crash:

        # First Time Through------------------------------------
        if first:
            rgb_display()
            first = False

        # DRAWING ITEMS------------------------------------
        for b in button_list:
            b.draw(screen)

        mouse_position()

        # EVENTS-------------------------------------------
        for event in pygame.event.get():

            # EXITING GAME------------------------------------
            if event.type == pygame.QUIT:
                crash = False


            # PRESSING BUTTONS------------------------------------
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if RGB_button.rect.collidepoint(pos):
                    rgb_display()
                if NDVI_button.rect.collidepoint(pos):
                    ndvi_display()
                if RXD_button.rect.collidepoint(pos):
                    rxd_display()
                if plus_button.rect.collidepoint(pos):
                    plus_parameter()
                if minus_button.rect.collidepoint(pos):
                    minus_parameter()





        pygame.display.update()
        clock.tick(60)