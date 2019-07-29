import random
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import *


# Code taken from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        size = len(self.memory)
        size = batch_size if size >= batch_size else size
        return random.sample(self.memory, size)

    def set_capacity(self, capacity):
        self.capacity = capacity

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)

        self.fc_adv1 = nn.Linear(2048, 512)
        self.dp_adv = nn.Dropout(p=0.1)
        self.fc_adv2 = nn.Linear(512, outputs)

        self.fc_val1 = nn.Linear(2048, 512)
        self.dp_val = nn.Dropout(p=0.1)
        self.fc_val2 = nn.Linear(512, 1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        val = F.leaky_relu(self.fc_val1(x))
        val = self.dp_val(val)
        val = self.fc_val2(val)

        adv = F.leaky_relu(self.fc_adv1(x))
        adv = self.dp_adv(adv)
        adv = self.fc_adv2(adv)

        return val + adv - adv.mean()
