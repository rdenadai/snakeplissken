#!/usr/bin/env python

import sys, pygame
from pygame.locals import *
import numpy as np


# obj = np.dtype([
#     ('speed', np.float32, 0.1),
#     ('size_x', np.int32, 10),
#     ('size_y', np.int32, 10),
# ])
#player = np.array([(0.1, 10, 10)], dtype=obj)


class Player():

    def __init__(self):
        self.speed = 0.1
        self.size_x = 10
        self.size_y = 10

    def draw(self, x, y):
        pygame.draw.rect(screen, white, (x, y, 10, 10))


class Food():

    def __init__(self):
        self.pos_x = np.random.randint(0, width - 10)
        self.pos_y = np.random.randint(0, height - 10)

    def draw(self):
        print(self.pos_x, self.pos_y)
        pygame.draw.rect(screen, crimson, (self.pos_x, self.pos_y, 10, 10))


if __name__ == '__main__':
    pygame.init()

    # Create the player
    player = Player()
    # Start food?
    food = None

    size = width, height = 640, 480
    speed = [2, 2]
    black = (0, 0, 0)
    crimson = (220, 20, 60)
    white = (255, 255, 255)

    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Paint")

    # Let's center the player
    x = width / 2 + player.size_x
    y = height / 2 + player.size_y
    # Loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        # Key movement
        pressed = pygame.key.get_pressed()
        if pressed[K_UP]: y -= player.speed
        if pressed[K_DOWN]: y += player.speed
        if pressed[K_LEFT]: x -= player.speed
        if pressed[K_RIGHT]: x += player.speed
        # Border
        if x <= 0: x = 0
        if x >= width-10: x = width - player.size_x
        if y <= 0: y = 0
        if y >= height-10: y = height - player.size_y

        # Clean screen
        screen.fill(black)
        if not food:
            food = Food()
            food.draw()

        # Draw player
        player.draw(x, y)
        pygame.display.update()