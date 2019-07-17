#!/usr/bin/env python

import sys, os
import time
import random
import pygame as pyg
from pygame.locals import *
import torch
from torch.optim import Adam, RMSprop
import torch.nn.functional as F
from configs import *
from utils.utilities import *
from ai.model import Transition


def draw_object(scr, color, position):
    pyg.draw.rect(scr, color, position)


def select_action(state, n_actions, steps_done):
    sample = np.random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor(
            [[random.randrange(n_actions)]], device=device, dtype=torch.long
        )


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
    # game steps
    game_steps = 0
    # the epoch we are running
    i_epoch = 0
    # number of games played
    n_game = 1
    # time between start and maximum time before reload some elements (in case apples)
    elapsed_time = 0
    # Action to be executed by the agent
    action = None
    # Train phase
    train, restart_mem, exploit = False, False, True

    # Screen size
    size = width, height = W_WIDTH, W_HEIGHT
    screen = pyg.display.set_mode(size, pyg.HWSURFACE)

    # Icon and Title
    pyg.display.set_icon(pyg.image.load("./img/snake.png"))
    pyg.display.set_caption("Snake Plissken")

    # print(get_game_screen(screen, device).shape)

    # Load model
    md_name = "snakeplissken_m2.model"
    policy_net, target_net, optimizer, memories = load_model(
        md_name, n_actions, device, restart_mem=restart_mem
    )
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Memory
    # Short is garbage
    short_memory = memories["short"]
    # Long is were the bad and good are
    good_long_memory = memories["good"]
    bad_long_memory = memories["bad"]

    # Game elements started
    t_score, score = 0, 0
    wall = get_walls(width, height)
    snake, apples = start_game(width, height)

    state, next_state = None, None
    t_start_game = time.time()

    # Game Main loop
    while True:
        start = time.time()
        for event in pyg.event.get():
            if event.type == pyg.QUIT:
                sys.exit()

        if train and steps_done % TARGET_UPDATE == 0:
            print("Update target network...")
            target_net.load_state_dict(policy_net.state_dict())
            memories = {
                "short": short_memory,
                "good": good_long_memory,
                "bad": bad_long_memory,
            }
            save_model(md_name, policy_net, target_net, optimizer, memories)

        # Stop the game, and restart
        if stop_game:
            print("-" * 20)
            print(
                f"Game end => score: {np.round(t_score, 5)}, steps: {steps}, game: {n_game}"
            )
            # Update the target network, copying all weights and biases in DQN
            if i_epoch % TARGET_UPDATE == 0 and train:
                print(f"Running for: {np.round(time.time() - t_start_game, 2)} secs")
                for param_group in optimizer.param_groups:
                    print(f"learning rate={param_group['lr']}")
            i_epoch += 1
            # Restart game elements
            state, next_state = None, None
            t_start_game = time.time()
            stop_game = False
            # Zeroed elapsed time
            elapsed_time = 0
            # Number of games +1
            n_game += 1
            game_steps = 0
            t_score, score = 0, 0
            snake, apples = start_game(width, height)

        # Load again the new screen: Initial State
        if state is None:
            state = get_game_screen(screen, device)

        # Action and reward of the agent
        if train and not exploit:
            action = select_action(state, n_actions, steps_done)
        else:
            with torch.no_grad():
                action = policy_net(state).max(1)[1].view(1, 1)

        # Key movements of agent to be done
        K = action.item()
        # if K == 1:
        #     if snake.head().direction == KEY["UP"]:
        #         snake.head().direction = KEY["LEFT"]
        #     elif snake.head().direction == KEY["LEFT"]:
        #         snake.head().direction = KEY["DOWN"]
        #     elif snake.head().direction == KEY["DOWN"]:
        #         snake.head().direction = KEY["RIGHT"]
        #     elif snake.head().direction == KEY["RIGHT"]:
        #         snake.head().direction = KEY["UP"]
        # elif K == 2:
        #     if snake.head().direction == KEY["UP"]:
        #         snake.head().direction = KEY["RIGHT"]
        #     elif snake.head().direction == KEY["RIGHT"]:
        #         snake.head().direction = KEY["DOWN"]
        #     elif snake.head().direction == KEY["DOWN"]:
        #         snake.head().direction = KEY["LEFT"]
        #     elif snake.head().direction == KEY["LEFT"]:
        #         snake.head().direction = KEY["UP"]

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
            score = -0.5
            stop_game = True

        # Wall collision
        # Check limits ! Border of screen
        for block in wall:
            if check_collision(snake.head(), block):
                score = -1.0
                stop_game = True
                break

        # Check collision between snake and apple
        del_apples = []
        for i, apple in enumerate(apples):
            if check_collision(snake.head(), apple):
                del_apples.append(i)
                score = APPLE_PRIZE
                t_score += APPLE_PRIZE
                snake.grow()
                break

        # Clean screen
        screen.fill(BLACK)

        # Draw snake
        for segment in reversed(snake.stack):
            draw_object(screen, segment.color, (segment.x, segment.y) + segment.size)

        # Draw Border
        for block in wall:
            draw_object(screen, block.color, block.position)

        # Draw appples
        if len(apples) == 0:
            apples = get_apples(width, height)
        for apple in apples:
            draw_object(screen, apple.color, apple.position)

        for i in del_apples:
            apples[i] = None
        apples = list(filter(None.__ne__, apples))

        # Print on the screen the score and other info
        steps = (
            f"{np.round(steps_done / 1000, 2)}k" if steps_done > 1000 else steps_done
        )
        # str_score = font.render(
        #     f"score: {np.round(score, 2)}, steps: {steps}, game: {n_game}", True, WHITE
        # )
        # screen.blit(str_score, (10, 10))

        # Next state for the agent
        next_state = get_game_screen(screen, device)
        # Give some points because it alive
        if not stop_game:
            score = -5e-3 if score == 0 else score
        else:
            next_state = None

        # if score == 1:
        #     save_game_screen("state.jpg", state)
        #     save_game_screen("next_state.jpg", next_state)
        #     print(score)
        #     break

        if train:
            reward = torch.tensor([score], device=device, dtype=torch.float)
            # Reward for the agent
            if not stop_game:
                if steps_done % 5 == 0:
                    # Store the transition in memory
                    short_memory.push(state, action, next_state, reward)
            else:
                # Store the transition in memory
                if score == APPLE_PRIZE:
                    good_long_memory.push(state, action, next_state, reward)
                else:
                    bad_long_memory.push(state, action, next_state, reward)
        # Move to the next state
        state = next_state
        score = 0

        # ----------------------------------------
        # Perform one step of the optimization (on the target network)
        if train and len(short_memory) > BATCH_SIZE:
            # for _ in range(2):
            transitions = []
            for memory in [short_memory, good_long_memory, bad_long_memory]:
                transitions += memory.sample(BATCH_SIZE)
            size = len(transitions)
            size = BATCH_SIZE if size > BATCH_SIZE else size
            transitions = random.sample(transitions, size)
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

            # Compute MSE loss
            # loss = F.mse_loss(
            #     state_action_values, expected_state_action_values.unsqueeze(1)
            # )
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
        game_steps += 1

        # Routines of pygame
        clock.tick(FPS)
        pyg.display.flip()
        pyg.display.update()

        # Reload apples position after some time
        elapsed_time += time.time() - start
        if elapsed_time > APPLE_RELOAD_TIME:
            elapsed_time = 0
            apples = get_apples(width, height)
