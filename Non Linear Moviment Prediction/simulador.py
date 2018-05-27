import pygame
import random
import numpy as np

display_size = 512

class Prjtl():
    
    def __init__(self, display_size):
        self.x = display_size/2
        self.y = display_size/2
        self.speed = random.random()*40+5
        self.direction_list = [-1, 1]
        self.coef_x = random.random()*random.choice(self.direction_list)
        self.coef_y = (1-abs(self.coef_x))*random.choice(self.direction_list)
        self.acceleration = -random.random()/3
        
    def update(self):
        if self.x < 0:
            self.x = 0
            self.coef_x *= -1

        if self.x > display_size:
            self.x = display_size
            self.coef_x *= -1

        if self.y < 0:
            self.y = 0
            self.coef_y *= -1

        if self.y > display_size:
            self.y = display_size
            self.coef_y *= -1

        if self.speed>=0:
            self.speed += self.acceleration
            self.x += self.speed*self.coef_x
            self.y += self.speed*self.coef_y
        
    def render(self):
        pygame.draw.circle(gameDisplay, white, (int(self.x), int(self.y)), 5)

        
    def get_position(self):
        return self.x, self.y


pygame.init()

black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 50, 50)
green = (50, 150, 50)
gameDisplay = pygame.display.set_mode((display_size, display_size))

pygame.display.set_caption('simulaçao')

clock = pygame.time.Clock()

crashed = False
fcounter = 0
repeat_counter = 0
mov_list = []
aux3 = []

frame_counter = 0
projetil = Prjtl(display_size)
while not crashed:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            crashed == True
        # print(event)

    gameDisplay.fill(green)
    projetil.update()
    projetil.render()
    pygame.display.update()
    clock.tick(60)
    frame_counter += 1
    aux1, aux2 = projetil.get_position()
    aux3 += [[aux1, aux2]]
    fcounter += 1
    if frame_counter >= 200:
        mov_list += [aux3]
        aux3 = []
        projetil = Prjtl(display_size)
        frame_counter = 0
        repeat_counter += 1
    if repeat_counter >= 200:
        print(mov_list)
        mov_list = np.array(mov_list)
        np.savetxt('data.txt', mov_list, fmt='%s',delimiter=',')
        print(mov_list.shape)
        break

# https://stackoverflow.com/questions/3685265/how-to-write-a-multidimensional-array-to-a-text-file
pygame.quit()
quit()

