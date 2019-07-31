#!/usr/bin/env python

import sys, os
import time
import random
import pygame as pyg
from pygame.locals import *
import torch
from torch.optim import Adam, RMSprop
from torch.optim.lr_scheduler import CyclicLR
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
    # number of games played
    n_game = 0
    # Action to be executed by the agent
    action = None
    # Train phase
    train, exploit, show_screen = True, False, True
    options = {
        "restart_mem": False,
        "restart_models": False,
        "restart_optim": False,
        "opt": "adam",
    }

    # Screen size
    size = width, height = W_WIDTH, W_HEIGHT
    screen = pyg.Surface(size)
    if show_screen:
        screen = pyg.display.set_mode(size, pyg.DOUBLEBUF)
        # Icon and Title
        pyg.display.set_icon(pyg.image.load("./img/snake.png"))
        pyg.display.set_caption("Snake Plissken")

    # print(get_game_screen(screen, device).shape)

    # Load model
    md_name = "snakeplissken_m2.model"
    policy_net, target_net, optimizer, memories = load_model(
        md_name, n_actions, device, **options
    )
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # _lr_scheduler = CyclicLR(
    #     optimizer,
    #     base_lr=LEARNING_RATE,
    #     max_lr=1e-5,
    #     step_size_up=5000,
    #     mode="exp_range",
    #     gamma=GAMMA,
    #     cycle_momentum=False,
    # )

    # Memory
    # Short is garbage
    short_memory = memories["short"]
    # Long is were the bad and good are
    good_long_memory = memories["good"]
    bad_long_memory = memories["bad"]

    vloss = [0]

    # Game elements started
    t_score, p_score, score = [], 0, 0
    wall = get_walls(width, height)
    snake, apples = start_game(width, height)

    state, next_state = None, None
    t_start_game = time.time()

    # Game Main loop
    while True:
        if show_screen:
            for event in pyg.event.get():
                if event.type == pyg.QUIT:
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
            p_score, score = 0, 0
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
                score = APPLE_PRIZE
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
            apples = get_apples(width, height)
        for apple in apples:
            draw_object(screen, apple.color, apple.position)

        for i in del_apples:
            apples[i] = None
        apples = list(filter(None.__ne__, apples))

        # Reload apples position after some time
        if steps_done % APPLE_RELOAD_STEPS == 0:
            apples = get_apples(width, height)

        # Print on the screen the score and other info
        # str_score = font.render(
        #     f"score: {np.round(score, 2)}, steps: {steps}, game: {n_game}", True, WHITE
        # )
        # screen.blit(str_score, (10, 10))

        # Next state for the agent
        next_state = get_game_screen(screen, device)
        # Give some points because it alive
        if not stop_game:
            score = SNAKE_ALIVE_PRIZE if score == 0 else score
        else:
            next_state = None

        if train:
            reward = torch.tensor([score], device=device, dtype=torch.float)
            # Reward for the agent
            if not stop_game:
                if score >= APPLE_PRIZE:
                    good_long_memory.push(state, action, next_state, reward)
                if score <= 0:
                    # Store the transition in memory
                    short_memory.push(state, action, next_state, reward)
            else:
                # Store the transition in memory
                bad_long_memory.push(state, action, next_state, reward)
        score = 0
        # Move to the next state
        state = next_state

        # ----------------------------------------
        # Perform one step of the optimization (on the target network)
        if train and len(short_memory) > (BATCH_SIZE):
            if steps_done % 10000 == 0:
                exploit = not exploit

            for param_group in optimizer.param_groups:
                if param_group["lr"] != LEARNING_RATE:
                    param_group["lr"] = LEARNING_RATE
                    break

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
                tuple(map(lambda s: s is not None, batch.next_state)), device=device
            )
            final_mask = 1 - non_final_mask

            non_final_next_states = torch.cat(
                [s for s in batch.next_state if s is not None]
            )
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            reward_batch.data.clamp_(-1, 1)
            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            state_action_values = policy_net(state_batch).gather(1, action_batch)

            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1)[0].
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            expected_state_action_values = torch.zeros(BATCH_SIZE, device=device)
            # Compute the expected Q values
            expected_state_action_values[non_final_mask] = (
                target_net(non_final_next_states).max(1)[0].detach() * GAMMA
                + reward_batch[non_final_mask].detach()
            )
            expected_state_action_values[final_mask] = reward_batch[final_mask].detach()

            # Compute MSE loss
            loss = F.mse_loss(
                state_action_values, expected_state_action_values.unsqueeze(1)
            )
            # Compute Huber loss
            # loss = F.smooth_l1_loss(
            #     state_action_values, expected_state_action_values.unsqueeze(1)
            # )
            vloss += [loss.item()]
            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            for param in policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()
            # _lr_scheduler.step()
        # ----------------------------------------

        # Routines of pygame
        clock.tick(FPS)
        if show_screen:
            pyg.display.flip()
            pyg.display.update()

        if train and steps_done % TARGET_UPDATE == 0:
            steps = (
                f"{np.round(steps_done / 1000, 2)}k"
                if steps_done > 1000
                else steps_done
            )
            print("*" * 20)
            print(f"Steps: {steps}, N Game: {n_game}")
            print(f"Score: {np.round(np.mean(t_score), 5)}")
            print(f"Running for: {np.round(time.time() - t_start_game, 2)} secs")
            print(f"In training mode: {train}")
            print(f"In exploit mode: {exploit}")
            print(f"Batch: {BATCH_SIZE}")
            print(f"Loss: {np.round(np.mean(vloss), 5)}")
            print("Optimizer:", optimizer.__class__.__name__)
            for param_group in optimizer.param_groups:
                print(f"learning rate={param_group['lr']}")
                break
            print("Memories:")
            print("  - short: ", len(memories["short"]))
            print("  - good: ", len(memories["good"]))
            print("  - bad: ", len(memories["bad"]))
            print("Update target network...")
            target_net.load_state_dict(policy_net.state_dict())
            memories = {
                "short": short_memory,
                "good": good_long_memory,
                "bad": bad_long_memory,
            }
            save_model(md_name, policy_net, target_net, optimizer, memories)
            t_score, vloss = [1], [0]

        # One step done in the whole game...
        steps_done += 1
