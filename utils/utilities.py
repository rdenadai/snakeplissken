import numpy as np
from numba import jit
import pygame
from PIL import Image
import torch
import torchvision.transforms as T
from configs import *
from objects.classes import Snake, Apple, Wall


@jit(parallel=True, nopython=True)
def random_position(x, y, width, height):
    x = np.random.choice(np.arange(x, width - x, 10))
    y = np.random.choice(np.arange(y, height - y, 10))
    return x, y


def check_collision(objA, objB, objA_size=SNAKE_SIZE, objB_size=APPLE_SIZE):
    if (
        objA.x <= objB.x + objB_size
        and objA.x + objA_size >= objB.x
        and objA.y <= objB.y + objB_size
        and objA.y + objA_size >= objB.y
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


def get_game_screen(screen, device):
    resize = T.Compose(
        [T.ToPILImage(), T.Resize(IMG_SIZE, interpolation=Image.BILINEAR), T.ToTensor()]
    )

    screen = np.rot90(pygame.surfarray.array3d(screen))[::-1]
    # screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.0
    # screen = torch.from_numpy(screen.transpose((2, 0, 1)))
    screen = resize(screen) / 255.0
    return screen.unsqueeze(0).to(device)


def save_game_screen(fname, img):
    Image.fromarray(img).save(fname)


def reload_apple(width, height):
    x, y = random_position(10, 10, width, height)
    return Apple(x, y, CRIMSON)


def get_apples(width, height):
    return [reload_apple(width, height) for _ in range(APPLE_QTD)]


def get_walls(width, height):
    x, y = 0, 0
    wall = (
        [Wall(x, 0, GRAY) for x in np.arange(0, width, 10)]
        + [Wall(x, height - 10, GRAY) for x in np.arange(0, width, 10)]
        + [Wall(0, y, GRAY) for y in np.arange(0, height, 10)]
        + [Wall(width - 10, y, GRAY) for y in np.arange(0, height, 10)]
    )
    return wall


def start_game(width, height):
    # Create the player
    x, y = random_position(20, 20, width, height)
    snake = Snake(x, y, GREEN, WHITE)
    # Start food?
    apples = get_apples(width, height)
    return snake, apples
