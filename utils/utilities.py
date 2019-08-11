import pickle
import numpy as np
from numba import jit
import pygame
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.utils import save_image
import torch.optim as optim
from configs import *
from objects.classes import Snake, Apple, Wall
from ai.model import DuelingDQN, ReplayMemory


@jit(parallel=True, nogil=True, nopython=True)
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


@jit(nopython=True, nogil=True, cache=True)
def check_object_collision(Ax, Ay, Bx, By, A_size=SNAKE_SIZE, B_size=APPLE_SIZE):
    if (
        Ax <= Bx + B_size
        and Ax + A_size >= Bx
        and Ay <= By + B_size
        and Ay + A_size >= By
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
    normalize = T.Compose(
        [
            T.ToPILImage(),
            T.Grayscale(1),
            T.Resize(IMG_SIZE, interpolation=Image.BILINEAR),
        ]
    )
    screen = np.rot90(pygame.surfarray.array3d(screen))[::-1]
    return np.array(normalize(screen), dtype=np.float32) / 255.0


def get_state(screen, device):
    screen = get_game_screen(screen, device)
    screen = np.stack([screen.astype(np.float32) for _ in range(4)], axis=0)
    screen = torch.from_numpy(screen)
    return screen.unsqueeze(0).to(device)


def get_next_state(screen, state, device):
    screen = get_game_screen(screen, device)
    states = state.cpu().squeeze(0)
    screen = np.stack([states[1, :], states[2, :], states[3, :], screen], axis=0)
    screen = torch.from_numpy(screen)
    return screen.unsqueeze(0).to(device)


def save_game_screen(fname, img):
    if img is not None:
        if isinstance(img, torch.Tensor):
            save_image(img.cpu().squeeze(0)[0], fname)
        else:
            Image.fromarray(img).save(fname)


def reload_apple(width, height, snake_pos):
    x, y = random_position(10, 10, width, height)
    collide = True
    while collide:
        check_collision = False
        for pos in snake_pos:
            if check_object_collision(pos[0], pos[1], x, y):
                check_collision = True
                break
        if check_collision:
            x, y = random_position(10, 10, width, height)
        else:
            collide = False
    return Apple(x, y, CRIMSON)


def get_apples(width, height, snake_pos):
    return [reload_apple(width, height, snake_pos) for _ in range(APPLE_QTD)]


def get_walls(width, height):
    x, y = 0, 0
    wall = (
        [Wall(x, 0, GRAY) for x in np.arange(0, width, 10)]
        + [Wall(x, height - 10, GRAY) for x in np.arange(0, width, 10)]
        + [Wall(0, y, GRAY) for y in np.arange(0, height, 10)]
        + [Wall(width - 10, y, GRAY) for y in np.arange(0, height, 10)]
    )
    return wall


def get_snake_position(snake):
    return [(segment.x, segment.y) for segment in snake.stack]


def start_game(width, height):
    # Create the player
    x, y = random_position(20, 20, width, height)
    snake = Snake(x, y, GREEN, WHITE)
    # Start food?
    apples = get_apples(width, height, get_snake_position(snake))
    return snake, apples


def save_model(name, policy_net, target_net, optimizer, memories):
    print("Saving model... wait...")
    torch.save(
        {
            "dqn": policy_net.state_dict(),
            "target": target_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "memories": memories,
        },
        name,
    )
    print("Model saved!")


def load_model(
    md_name,
    n_actions,
    device,
    restart_mem=False,
    restart_models=False,
    restart_optim=False,
    random_clean_memory=False,
    opt="adam",
):
    # DuelingDQN Algoritm
    policy_net = DuelingDQN(n_actions).to(device)
    target_net = DuelingDQN(n_actions).to(device)
    # Optimizer
    if "adam" in opt:
        optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    elif "rmsprop" in opt:
        optimizer = optim.RMSprop(
            policy_net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM
        )
    elif "sgd" in opt:
        optimizer = optim.SGD(
            policy_net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM
        )

    memories = {
        "short": ReplayMemory(int(MEM_LENGTH * 2.55)),
        "good": ReplayMemory(int(MEM_LENGTH * 2.55)),
        "bad": ReplayMemory(int(MEM_LENGTH * 1.55)),
    }

    try:
        checkpoint = torch.load(md_name, map_location=device)
        if not restart_models:
            policy_net.load_state_dict(checkpoint["dqn"])
            target_net.load_state_dict(checkpoint["target"])
        if not restart_optim:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if not restart_mem:
            memories = checkpoint["memories"]
            memories["short"].set_capacity(int(MEM_LENGTH * 2.55))
            memories["good"].set_capacity(int(MEM_LENGTH * 2.55))
            memories["bad"].set_capacity(int(MEM_LENGTH * 1.55))
            if random_clean_memory:
                memories["short"].random_clean_memory(MEM_CLEAN_SIZE)
                memories["good"].random_clean_memory(MEM_CLEAN_SIZE)
                memories["bad"].random_clean_memory(MEM_CLEAN_SIZE)
        print("Models loaded!")
    except Exception as e:
        print(f"Couldn't load Models! => {e}")
    return policy_net, target_net, optimizer, memories


def load_model_only(md_name, n_actions, device):
    policy_net = DuelingDQN(n_actions).to(device)
    target_net = DuelingDQN(n_actions).to(device)
    checkpoint = torch.load(md_name, map_location=device)
    policy_net.load_state_dict(checkpoint["dqn"])
    target_net.load_state_dict(checkpoint["target"])
    return policy_net, target_net


def save_model_only(md_name, n_actions, device):
    policy_net = DuelingDQN(n_actions).to(device)
    target_net = DuelingDQN(n_actions).to(device)
    checkpoint = torch.load(md_name, map_location=device)
    policy_net.load_state_dict(checkpoint["dqn"])
    target_net.load_state_dict(checkpoint["target"])
    torch.save(
        {"dqn": policy_net.state_dict(), "target": target_net.state_dict()}, md_name
    )
