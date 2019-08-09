#!/usr/bin/env python

import sys, os
import time
import random
import pygame as pyg
from pygame.locals import *
import torch
from configs import *
from utils.utilities import *


def draw_object(scr, color, position):
    pyg.draw.rect(scr, color, position)


if __name__ == "__main__":
    # In linux center the window
    os.environ["SDL_VIDEO_CENTERED"] = "1"

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Pygame init loop
    pyg.init()

    # confs for pygame
    stop_game = False
    clock = pyg.time.Clock()
    font = pyg.font.Font(None, 20)
    # number o actions the agent can do
    n_actions = 4
    # number of steps done, each step is a run in while loop
    steps_done = 0
    # number of games played
    n_game = 0
    # Action to be executed by the agent
    action = None
    # Screen size
    size = width, height = W_WIDTH, W_HEIGHT
    screen = pyg.display.set_mode(size, pyg.DOUBLEBUF)
    # Icon and Title
    pyg.display.set_icon(pyg.image.load("./img/snake.png"))
    pyg.display.set_caption("Snake Plissken")

    # Load model
    md_name = "snakeplissken_m2.model"
    policy_net, target_net = load_model_only(md_name, n_actions, device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Game elements started
    t_score, p_score, score = [1], 0, 0
    wall = get_walls(width, height)
    snake, apples = start_game(width, height)

    state, next_state = None, None
    t_start_game = time.time()

    # Game Main loop
    while True:
        for event in pyg.event.get():
            if event.type == pyg.QUIT:
                pyg.quit()
                sys.exit()

        # Stop the game, and restart
        if stop_game:
            # Restart game elements
            state, next_state = None, None
            stop_game = False
            # Zeroed elapsed time
            elapsed_time = 0
            # Number of games +1
            n_game += 1
            t_score += [p_score]
            print(f"Score : {p_score}")
            p_score, score = 0, 0
            snake, apples = start_game(width, height)

        # Load again the new screen: Initial State
        if state is None:
            state = get_state(screen, device)

        # Action and reward of the agent
        with torch.no_grad():
            action = policy_net(state).max(1)[1].view(1, 1)

        # Key movements of agent to be done
        K = action.item()
        if K == 0 and snake.head().direction != KEY["DOWN"]:
            snake.head().direction = KEY["UP"]
        elif K == 1 and snake.head().direction != KEY["UP"]:
            snake.head().direction = KEY["DOWN"]
        elif K == 2 and snake.head().direction != KEY["RIGHT"]:
            snake.head().direction = KEY["LEFT"]
        elif K == 3 and snake.head().direction != KEY["LEFT"]:
            snake.head().direction = KEY["RIGHT"]

        # Human keys!
        # pressed = pyg.key.get_pressed()
        # if pressed[K_UP] and snake.head().direction != KEY["DOWN"]:
        #     snake.head().direction = KEY["UP"]
        # elif pressed[K_DOWN] and snake.head().direction != KEY["UP"]:
        #     snake.head().direction = KEY["DOWN"]
        # elif pressed[K_LEFT] and snake.head().direction != KEY["RIGHT"]:
        #     snake.head().direction = KEY["LEFT"]
        # elif pressed[K_RIGHT] and snake.head().direction != KEY["LEFT"]:
        #     snake.head().direction = KEY["RIGHT"]

        # Move of snake...
        snake.move()

        # Snake crash to its tail
        if check_crash(snake):
            score = SNAKE_EAT_ITSELF_PRIZE + sum([1e-3 for segment in snake.stack])
            stop_game = True

        # Wall collision
        # Check limits ! Border of screen
        for block in wall:
            if check_collision(snake.head(), block):
                score = WALL_PRIZE
                stop_game = True
                break

        # Check collision between snake and apple
        del_apples = []
        for i, apple in enumerate(apples):
            if check_collision(snake.head(), apple):
                del_apples.append(i)
                p_score += APPLE_PRIZE
                score = APPLE_PRIZE + sum([1e-3 for segment in snake.stack])
                snake.grow()
                break

        # Clean screen
        screen.fill(BLACK)

        # Draw Border
        for block in wall:
            draw_object(screen, block.color, block.position)

        # Draw snake
        for segment in snake.stack:
            draw_object(screen, segment.color, (segment.x, segment.y) + segment.size)

        # Draw appples
        if len(apples) == 0:
            apples = get_apples(width, height, get_snake_position(snake))
        for apple in apples:
            draw_object(screen, apple.color, apple.position)

        for i in del_apples:
            apples[i] = None
        apples = list(filter(None.__ne__, apples))

        # Reload apples position after some time
        if steps_done % APPLE_RELOAD_STEPS == 0:
            apples = get_apples(width, height, get_snake_position(snake))

        # Print on the screen the score and other info
        # str_score = font.render(
        #     f"score: {np.round(score, 2)}, steps: {steps}, game: {n_game}", True, WHITE
        # )
        # screen.blit(str_score, (10, 10))

        # Next state for the agent
        next_state = None
        # Give some points because it alive
        if not stop_game:
            score = SNAKE_ALIVE_PRIZE if score == 0 else score
            next_state = get_next_state(screen, state, device)

        score = 0
        # Move to the next state
        state = next_state

        # Routines of pygame
        clock.tick(FPS_PLAY)
        pyg.display.update()

        # One step done in the whole game...
        steps_done += 1
