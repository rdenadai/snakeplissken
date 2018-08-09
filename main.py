#!/usr/bin/env python

import sys, pygame
from pygame.locals import *
import numpy as np
from objects.configs import FPS, BLACK, WHITE, CRIMSON, GREEN, KEY, SNAKE_SIZE, APPLE_SIZE
from objects.classes import Snake, Apple


def check_collision(objA, objB, objA_size=SNAKE_SIZE, objB_size=APPLE_SIZE):
    if(
        objA.x < objB.x + objB_size and
        objA.x + objA_size > objB.x and
        objA.y < objB.y + objB_size and
        objA.y + objA_size > objB.y
    ):
        return True
    return False


def check_crash(snake):
    counter = 1
    stack = snake.stack
    while counter < len(stack) - 1:
        if check_collision(stack[0], stack[counter], SNAKE_SIZE, SNAKE_SIZE):
            return True
        counter += 1
    return False


def draw_object(scr, color, position):
    pygame.draw.rect(scr, color, position)


def reload_apple():
    apple_x = np.random.choice(np.arange(10, width - 10, 10))
    apple_y = np.random.choice(np.arange(10, height - 10, 10))
    return Apple(apple_x, apple_y, CRIMSON)


def start_game():
    score = 0
    # Create the player
    x = np.random.choice(np.arange(80, width - 80, 10))
    y = np.random.choice(np.arange(80, height - 80, 10))
    snake = Snake(x, y, GREEN, WHITE)
    # Start food?
    apple = reload_apple()
    return score, snake, apple


if __name__ == '__main__':
    pygame.init()
    stop_game = False
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 20)

    size = width, height = 640, 480

    screen = pygame.display.set_mode(size, pygame.HWSURFACE)
    pygame.display.set_caption("Snake Plissken")

    score, snake, apple = start_game()   
    # Loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        if stop_game:
            score, snake, apple = start_game()
            stop_game = False

        # Key movement
        pressed = pygame.key.get_pressed()
        if pressed[K_UP]: snake.head().direction = KEY['UP']
        if pressed[K_DOWN]: snake.head().direction = KEY['DOWN']
        if pressed[K_LEFT]: snake.head().direction = KEY['LEFT']
        if pressed[K_RIGHT]: snake.head().direction = KEY['RIGHT']

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
            stop_game = True

        # Clean screen
        screen.fill(BLACK)

        # Draw appple
        if not apple:
            apple = reload_apple()
        draw_object(screen, apple.color, apple.position)

        # Draw snake
        snake.move()
        for segment in snake.stack:
            draw_object(screen, segment.color, (segment.x, segment.y) + segment.size)

        # Check collision between snake and apple
        if check_collision(snake.head(), apple):
            apple = None
            score += 1
            snake.grow()

        # Print on the screen the score
        str_score = font.render(f'score: {score}', True, WHITE)
        screen.blit(str_score, (5, 5))

        # Routines
        clock.tick(FPS)
        pygame.display.flip()
        pygame.display.update()
