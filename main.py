#!/usr/bin/env python

import sys, os
import pygame as pyg
from pygame.locals import *
import torch
import torch.optim as optim
from configs import *
from utils.utilities import *
from ai.model import DQN, ReplayMemory, select_action, optimize_model


def draw_object(scr, color, position):
    pyg.draw.rect(scr, color, position)


if __name__ == "__main__":
    # In linux center the window
    os.environ["SDL_VIDEO_CENTERED"] = "1"

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pygame init loop
    pyg.init()

    # confs
    stop_game = False
    clock = pyg.time.Clock()
    font = pyg.font.Font(None, 20)
    n_actions = 4
    steps_done = 0
    action = None

    # Screen size
    size = width, height = W_WIDTH, W_HEIGHT
    screen = pyg.display.set_mode(size, pyg.HWSURFACE)

    # Icon and Title
    pyg.display.set_icon(pyg.image.load("./img/snake.png"))
    pyg.display.set_caption("Snake Plissken")

    # DQN Algoritm
    policy_net = DQN(80, 106, n_actions).to(device)
    target_net = DQN(80, 106, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    # Optimizer
    optimizer = optim.RMSprop(policy_net.parameters())
    # Memory
    memory = ReplayMemory(10000)

    score, snake, apples = start_game(width, height)
    # Game Main loop
    while True:
        for event in pyg.event.get():
            if event.type == pyg.QUIT:
                sys.exit()

        # Load image screen data as torch Tensor : State
        last_screen = get_game_screen(screen, device)
        current_screen = get_game_screen(screen, device)
        state = current_screen - last_screen
        action = select_action(device, state, n_actions, steps_done, policy_net)
        reward = torch.tensor([score], device=device, dtype=torch.float)

        # Stop the game, and restart
        if stop_game:
            score, snake, apples = start_game(width, height)
            stop_game = False

        # Key movement
        K = action.item()
        if K == 1 and snake.head().direction != KEY["DOWN"]:
            snake.head().direction = KEY["UP"]
        if K == 2 and snake.head().direction != KEY["UP"]:
            snake.head().direction = KEY["DOWN"]
        if K == 3 and snake.head().direction != KEY["RIGHT"]:
            snake.head().direction = KEY["LEFT"]
        if K == 4 and snake.head().direction != KEY["LEFT"]:
            snake.head().direction = KEY["RIGHT"]

        # pressed = pyg.key.get_pressed()
        # if pressed[K_UP] and snake.head().direction != KEY["DOWN"]:
        #     snake.head().direction = KEY["UP"]
        # if pressed[K_DOWN] and snake.head().direction != KEY["UP"]:
        #     snake.head().direction = KEY["DOWN"]
        # if pressed[K_LEFT] and snake.head().direction != KEY["RIGHT"]:
        #     snake.head().direction = KEY["LEFT"]
        # if pressed[K_RIGHT] and snake.head().direction != KEY["LEFT"]:
        #     snake.head().direction = KEY["RIGHT"]

        # Check limits ! Border
        boundary_hit = False
        if snake.head().x <= 0:
            boundary_hit = True
        if snake.head().x >= width:
            boundary_hit = True
        if snake.head().y <= 0:
            boundary_hit = True
        if snake.head().y >= height - 10:
            boundary_hit = True
        if boundary_hit:
            score -= 200
            stop_game = True

        # Snake crash to its tail
        if check_crash(snake):
            score -= 100
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
                score += 5
                snake.grow()
        # Clear empty apples
        apples = list(filter(None.__ne__, apples))

        # Print on the screen the score
        str_score = font.render(f"score: {score}, epoch: {steps_done}", True, WHITE)
        screen.blit(str_score, (5, 5))

        # Routines
        clock.tick(FPS)
        pyg.display.flip()
        pyg.display.update()

        last_screen = current_screen
        current_screen = get_game_screen(screen, device)
        next_state = current_screen - last_screen
        # Store the transition in memory
        memory.push(state, action, next_state, reward)
        # Move to the next state
        state = next_state
        # Perform one step of the optimization (on the target network)
        optimize_model(device, policy_net, target_net, memory, optimizer)
        # Update the target network, copying all weights and biases in DQN
        if steps_done % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        steps_done += 1

