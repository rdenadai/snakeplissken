#!/usr/bin/env python

import sys, os
import pygame
from pygame.locals import *
from PIL import Image
from objects.configs import *
from utils.utilities import *


def draw_object(scr, color, position):
    pygame.draw.rect(scr, color, position)


if __name__ == '__main__':
    # In linux center the window
    os.environ['SDL_VIDEO_CENTERED'] = '1'

    # Pygame init loop
    pygame.init()

    # confs
    stop_game = False
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 20)

    # Screen size
    size = width, height = W_WIDTH, W_HEIGHT
    screen = pygame.display.set_mode(size, pygame.HWSURFACE)

    # Icon and Title
    pygame.display.set_icon(pygame.image.load('./img/snake.png'))
    pygame.display.set_caption("Snake Plissken")

    score, snake, apples = start_game(width, height)
    # Game Main loop
    while True:
        img = pygame.surfarray.array3d(screen)[::-1]
        im = Image.fromarray(img)
        im.save("/home/rdenadai/your_file.png")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        # Stop the game, and restart
        if stop_game:
            score, snake, apple = start_game(width, height)
            stop_game = False

        # Key movement
        pressed = pygame.key.get_pressed()
        if pressed[K_UP] and snake.head().direction != KEY['DOWN']: snake.head().direction = KEY['UP']
        if pressed[K_DOWN] and snake.head().direction != KEY['UP']: snake.head().direction = KEY['DOWN']
        if pressed[K_LEFT] and snake.head().direction != KEY['RIGHT']: snake.head().direction = KEY['LEFT']
        if pressed[K_RIGHT] and snake.head().direction != KEY['LEFT']: snake.head().direction = KEY['RIGHT']

        # Check limits ! Border
        boundary_hit = False
        if snake.head().x <= 0: boundary_hit = True
        if snake.head().x >= width: boundary_hit = True
        if snake.head().y <= 0: boundary_hit = True
        if snake.head().y >= height-10: boundary_hit = True
        if boundary_hit:
            score -= 1
            stop_game = True

        # Snake crash to its tail
        if check_crash(snake):
            score -= 1
            apples = get_apples(width, height)
            stop_game = True

        # Clean screen
        screen.fill(BLACK)

        # Draw appple
        if len(apples) == 0:
            apples = get_apples(width, height)
        for apple in apples:
            draw_object(screen, apple.color, apple.position)

        # Draw snake
        snake.move()
        for segment in snake.stack:
            draw_object(screen, segment.color, (segment.x, segment.y) + segment.size)

        # Check collision between snake and apple
        for i, apple in enumerate(apples):
            if check_collision(snake.head(), apple):
                apples[i] = None
                score += 1
                snake.grow()
        # Clear empty apples
        apples = list(filter(None.__ne__, apples))

        # Print on the screen the score
        str_score = font.render(f'score: {score}', True, WHITE)
        screen.blit(str_score, (5, 5))

        # Routines
        clock.tick(FPS)
        pygame.display.flip()
        pygame.display.update()
