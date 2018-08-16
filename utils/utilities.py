from objects.configs import *
from objects.classes import Snake, Apple
import numpy as np
from numba import jit


@jit
def check_collision(objA, objB, objA_size=SNAKE_SIZE, objB_size=APPLE_SIZE):
    if(
        objA.x < objB.x + objB_size and
        objA.x + objA_size > objB.x and
        objA.y < objB.y + objB_size and
        objA.y + objA_size > objB.y
    ):
        return True
    return False


@jit
def check_crash(snake):
    counter = 1
    stack = snake.stack
    while counter < len(stack) - 1:
        if check_collision(stack[0], stack[counter], SNAKE_SIZE, SNAKE_SIZE):
            return True
        counter += 1
    return False


def get_apples(width, height):
    return [reload_apple(width, height) for _ in range(APPLE_QTD)]


@jit
def reload_apple(width, height):
    apple_x = np.random.choice(np.arange(10, width - 10, 10))
    apple_y = np.random.choice(np.arange(10, height - 10, 10))
    return Apple(apple_x, apple_y, CRIMSON)


@jit
def start_game(width, height):
    score = 0
    # Create the player
    x = np.random.choice(np.arange(80, width - 80, 10))
    y = np.random.choice(np.arange(80, height - 80, 10))
    snake = Snake(x, y, GREEN, WHITE)
    # Start food?
    apples = get_apples(width, height)
    return score, snake, apples