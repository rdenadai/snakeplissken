#!/usr/bin/env python

import sys, os
import time
import pygame as pyg
from pygame.locals import *
import torch
import torch.optim as optim
import torch.nn.functional as F
from configs import *
from utils.utilities import *
from ai.model import DQN, ReplayMemory, Transition, select_action


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
    # the epoch we are running
    i_epoch = 0
    # number of games played
    n_game = 0
    # time between start and maximum time before reload some elements (in case apples)
    elapsed_time = 0
    # Action to be executed by the agent
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
    memory = ReplayMemory(25000)

    # Load image screen data as torch Tensor : Initial State
    last_screen = get_game_screen(screen, device)
    current_screen = get_game_screen(screen, device)
    state = current_screen - last_screen

    # Game elements started
    score, snake, apples = start_game(width, height)

    # Game Main loop
    while True:
        start = time.time()
        for event in pyg.event.get():
            if event.type == pyg.QUIT:
                sys.exit()

        # Stop the game, and restart
        if stop_game:
            # Zeroed elapsed time
            elapsed_time = 0
            # Number of games +1
            n_game += 1
            # Update the target network, copying all weights and biases in DQN
            if i_epoch % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
                i_epoch += 1
            # Load again the new screen: Initial State
            last_screen = get_game_screen(screen, device)
            current_screen = get_game_screen(screen, device)
            state = current_screen - last_screen
            # Restart game elements
            score, snake, apples = start_game(width, height)
            stop_game = False

        # Action and reward of the agent
        action = select_action(device, state, n_actions, steps_done, policy_net)

        # Key movements of agent to be done
        K = action.item()
        if K == 0 and snake.head().direction != KEY["DOWN"]:
            snake.head().direction = KEY["UP"]
        if K == 1 and snake.head().direction != KEY["UP"]:
            snake.head().direction = KEY["DOWN"]
        if K == 2 and snake.head().direction != KEY["RIGHT"]:
            snake.head().direction = KEY["LEFT"]
        if K == 3 and snake.head().direction != KEY["LEFT"]:
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

        # Check limits ! Border of screen
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

        # Draw appples
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

        # Print on the screen the score and other info
        steps = f"{np.round(steps_done / 1000)}k" if steps_done > 1000 else steps_done
        str_score = font.render(
            f"score: {round(score)}, steps: {steps}, game: {n_game}", True, WHITE
        )
        screen.blit(str_score, (5, 5))

        # Routines of pygame
        clock.tick(FPS)
        pyg.display.flip()
        pyg.display.update()

        # Add to score some minor points for being alive!
        score += 1e-5
        # Reward for the agent
        reward = torch.tensor([score], device=device, dtype=torch.float)
        # Next state for the agent
        last_screen = current_screen
        current_screen = get_game_screen(screen, device)
        next_state = current_screen - last_screen
        # Store the transition in memory
        memory.push(state, action, next_state, reward)
        # Move to the next state
        state = next_state

        # ----------------------------------------
        # Perform one step of the optimization (on the target network)
        if len(memory) >= BATCH_SIZE:
            transitions = memory.sample(BATCH_SIZE)
            # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
            # detailed explanation). This converts batch-array of Transitions
            # to Transition of batch-arrays.
            batch = Transition(*zip(*transitions))

            # Compute a mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)
            non_final_mask = torch.tensor(
                tuple(map(lambda s: s is not None, batch.next_state)),
                device=device,
                dtype=torch.uint8,
            )
            non_final_next_states = torch.cat(
                [s for s in batch.next_state if s is not None]
            )
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            state_action_values = policy_net(state_batch).gather(1, action_batch)

            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1)[0].
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            next_state_values = torch.zeros(BATCH_SIZE, device=device)
            next_state_values[non_final_mask] = (
                target_net(non_final_next_states).max(1)[0].detach()
            )
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * GAMMA) + reward_batch

            # Compute Huber loss
            loss = F.smooth_l1_loss(
                state_action_values, expected_state_action_values.unsqueeze(1)
            )

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            for param in policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()
        # ----------------------------------------

        # One step done in the whole game...
        steps_done += 1

        # Reload apples position after some time
        elapsed_time += time.time() - start
        if elapsed_time > APPLE_RELOAD_TIME:
            elapsed_time = 0
            apples = get_apples(width, height)

