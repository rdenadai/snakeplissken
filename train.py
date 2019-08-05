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


if __name__ == "__main__":
    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_actions = 4
    # number of steps done, each step is a run in while loop
    steps_done = 0
    # time between start and maximum time before reload some elements (in case apples)
    elapsed_time = 0
    # Train phase
    options = {
        "restart_mem": False,
        "restart_models": False,
        "restart_optim": False,
        "random_clean_memory": False,
        "opt": "adam",
    }

    # Load model
    md_name = "snakeplissken_m2.model"
    policy_net, target_net, optimizer, memories = load_model(
        md_name, n_actions, device, **options
    )
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Memory
    # Short is garbage
    short_memory = memories["short"]
    # Long is were the bad and good are
    good_long_memory = memories["good"]
    bad_long_memory = memories["bad"]

    vloss = [0]

    t_start_game = time.time()
    # Game Main loop
    for epoch in range(EPOCHS):
        if len(short_memory) > (BATCH_SIZE):
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
            # loss = F.mse_loss(
            #     state_action_values, expected_state_action_values.unsqueeze(1)
            # )
            # Compute Huber loss
            loss = F.smooth_l1_loss(
                state_action_values, expected_state_action_values.unsqueeze(1)
            )
            vloss += [loss.item()]
            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            for param in policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()
        # ----------------------------------------

        if steps_done % TARGET_UPDATE == 0:
            steps = (
                f"{np.round(steps_done / 1000, 2)}k"
                if steps_done > 1000
                else steps_done
            )
            print("*" * 20)
            print(f"Steps: {steps}")
            print(f"Running for: {np.round(time.time() - t_start_game, 2)} secs")
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
            vloss = [0]

        if steps_done % MODEL_SAVE == 0:
            memories = {
                "short": short_memory,
                "good": good_long_memory,
                "bad": bad_long_memory,
            }
            save_model(md_name, policy_net, target_net, optimizer, memories)
        # One step done in the whole game...
        steps_done += 1
