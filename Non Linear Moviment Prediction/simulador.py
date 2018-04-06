import pygame
import random

class Prjtl():
    
    def __init__(self):
        self.x = 512/2
        self.y = 512/2
        self.speed = random.random()*40+5
        self.direction_list = [-1, 1]
        self.coef_x = random.random()*random.choice(self.direction_list)
        self.coef_y = (1-abs(self.coef_x))*random.choice(self.direction_list)
        self.acceleration = -random.random()/3
        
    def update(self):
        if self.x > 512 or self.x < 0:
            self.coef_x *= -1
        if self.y > 512 or self.y < 0:
            self.coef_y *= -1
        if self.speed>=0:
            self.speed += self.acceleration
            self.x += self.speed*self.coef_x
            self.y += self.speed*self.coef_y
        
    def render(self):
        pygame.draw.circle(gameDisplay, red, (int(self.x), int(self.y)), 5)

        
    def get_position(self):
        return self.x, self.y


pygame.init()


display_size = 512

black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 50, 50)
green = (50, 255, 50)
gameDisplay = pygame.display.set_mode((display_size, display_size))

pygame.display.set_caption('simulaçao')

clock = pygame.time.Clock()

crashed = False

projetil = Prjtl()
while not crashed:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            crashed == True
        # print(event)

    gameDisplay.fill(black)
    projetil.update()
    projetil.render()
    pygame.display.update()
    clock.tick(60)

pygame.quit()
quit()

