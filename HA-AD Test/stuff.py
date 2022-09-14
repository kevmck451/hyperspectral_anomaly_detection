import pygame

class image:
    def __init__(self, image, position, scale, mid=True):
        self.image = image
        im_wid = self.image.get_width()
        im_hi = self.image.get_height()
        size = (im_wid * scale, im_hi * scale)
        self.image = pygame.transform.scale(image, size)
        self.surf = pygame.Surface(size)
        if mid:
            self.rect = self.surf.get_rect(center=position)
        else:
            self.rect = self.surf.get_rect(topleft=position)

    def draw(self, screen):
        screen.blit(self.image, self.rect)

class button:
    def __init__(self, position, size, clr=[100, 100, 100], cngclr=None, text='', font="Segoe Print", font_size=16, font_clr=[0, 0, 0]):
        self.clr    = clr
        self.size   = size
        self.surf   = pygame.Surface(size)
        self.rect   = self.surf.get_rect(center=position)

        if cngclr: self.cngclr = cngclr
        else: self.cngclr = clr

        if len(clr) == 4: self.surf.set_alpha(clr[3])

        self.font = pygame.font.SysFont(font, font_size)
        self.txt = text
        self.font_clr = font_clr
        self.txt_surf = self.font.render(self.txt, 1, self.font_clr)
        self.txt_rect = self.txt_surf.get_rect(center=[wh//2 for wh in self.size])

    def draw(self, screen):
        self.mouseover()

        self.surf.fill(self.curclr)
        self.surf.blit(self.txt_surf, self.txt_rect)
        screen.blit(self.surf, self.rect)

    def mouseover(self):
        self.curclr = self.clr
        pos = pygame.mouse.get_pos()
        if self.rect.collidepoint(pos):
            self.curclr = self.cngclr


class text:
    def __init__(self, msg, position, clr=[100, 100, 100], font="Segoe Print", font_size=15, mid=False):
        self.position = position
        self.font = pygame.font.SysFont(font, font_size)
        self.txt_surf = self.font.render(msg, 1, clr)

        if len(clr) == 4:
            self.txt_surf.set_alpha(clr[3])

        if mid:
            self.position = self.txt_surf.get_rect(center=position)


    def draw(self, screen):
        screen.blit(self.txt_surf, self.position)



# SCREEN CLASS FOR WINDOW HAVING THE FUNCTION
# OF UPDATING THE ONE SCREEN TO ANOTHER SCREEN

class Screen:
    # HERE (0,0,255) IS A COLOUR CODE
    def __init__(self, title, width=440, height=445, fill=(0, 0, 255)):
        # HEIGHT OF A WINDOW
        self.height = height
        # TITLE OF A WINDOW
        self.title = title
        # WIDTH OF A WINDOW
        self.width = width
        # COLOUR CODE
        self.fill = fill
        # CURRENT STATE OF A SCREEN
        self.CurrentState = False

    # DISPLAY THE CURRENT SCREEN OF
    # A WINDOW AT THE CURRENT STATE
    def makeCurrentScreen(self):
        # SET THE TITLE FOR THE CURRENT STATE OF A SCREEN
        pygame.display.set_caption(self.title)
        # SET THE STATE TO ACTIVE
        self.CurrentState = True
        # ACTIVE SCREEN SIZE
        self.screen = pygame.display.set_mode((self.width,
                                        self.height))

    # THIS WILL SET THE STATE OF A CURRENT STATE TO OFF

    def endCurrentScreen(self):
        self.CurrentState = False

    # THIS WILL CONFIRM WHETHER THE NAVIGATION OCCURS
    def checkUpdate(self, fill):
        self.fill = fill # HERE FILL IS THE COLOR CODE
        return self.CurrentState

    # THIS WILL UPDATE THE SCREEN WITH THE NEW NAVIGATION TAB
    def screenUpdate(self):
        if self.CurrentState:
                # IT WILL UPDATE THE COLOR OF THE SCREEN
            self.screen.fill(self.fill)

    # RETURNS THE TITLE OF THE SCREEN
    def returnTitle(self):
        return self.screen

