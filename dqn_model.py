import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Credit to https://gist.github.com/Jawdeyu/bf12742342fb65bbde687fa091e7da4e


class DQNNetwork(nn.Module):
    def __init__(self, input_shape, num_outputs):
        # input shape is (channels, height, width)
        super(DQNNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 8, kernel_size=3, stride=1), nn.BatchNorm2d(8)
        )

        # to calculate the output size of the last conv layer
        conv_out_size = self._get_conv_flat(input_shape)

        self.fc = nn.Sequential(nn.ReLU(), nn.Linear(conv_out_size, num_outputs))

    def _get_conv_flat(self, input_shape):
        out = self.conv(torch.zeros(1, *input_shape))
        return int(np.prod(out.size()))

    def forward(self, x):
        # x = x.to(self.device)
        conv_out = self.conv(x).view(x.size(0), -1)
        return self.fc(conv_out)


class DQNAgent(nn.Module):
    def __init__(
        self,
        state_space,
        action_space,
        max_memory_size,
        batch_size,
        gamma,
        lr,
        dropout,
        exploration_max,
        exploration_min,
        exploration_decay,
        pretrained,
    ):
        super(DQNAgent, self).__init__()
        # TODO make all these params actually used (e.g. exploration_decay)
        # DQN parameters
        self.state_space = state_space
        self.action_space = action_space
        self.pretrained = pretrained
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # DQN itself
        self.dqn = DQNNetwork(self.state_space, self.action_space).to(self.device)

        if self.pretrained:
            self.dqn.load_state_dict(
                torch.load("DQN.pt", map_location=torch.device(self.device))
            )

        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=lr)

        # Replay memory
        self.max_memory_size = max_memory_size
        if self.pretrained:
            self.STATE_MEM = torch.load("STATE_PREV_MEM.pt")
            self.ACTION_MEM = torch.load("ACTION_MEM.pt")
            self.REWARD_MEM = torch.load("REWARD_MEM.pt")
            self.STATE2_MEM = torch.load("STATE_NEXT_MEM.pt")
            self.DONE_MEM = torch.load("DONE_MEM.pt")

            with open("ending_position.pkl", "rb") as f:
                self.ending_position = pickle.load(f)
            with open("num_in_queue.pkl", "rb") as f:
                self.num_in_queue = pickle.load(f)

        else:
            self.STATE_PREV_MEM = torch.zeros(max_memory_size, *self.state_space)
            self.ACTION_MEM = torch.zeros(max_memory_size, 1)
            self.REWARD_MEM = torch.zeros(max_memory_size, 1)
            self.STATE_NEXT_MEM = torch.zeros(max_memory_size, *self.state_space)
            self.DONE_MEM = torch.zeros(max_memory_size, 1)
            self.ending_position = 0
            self.num_in_queue = 0

        # how much we sample from our replay memory each time
        self.memory_sample_size = batch_size

        # Learning params
        self.gamma = gamma
        self.dropout = dropout  # TODO add dropout later
        self.l1 = nn.SmoothL1Loss().to(self.device)  # Also known as Huber loss
        self.exploration_max = exploration_max
        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay

    def remember(self, state_prev, action, reward, state_next, done):
        self.STATE_PREV_MEM[self.ending_position] = state_prev.float()
        self.ACTION_MEM[self.ending_position] = action.float()
        self.REWARD_MEM[self.ending_position] = reward.float()
        self.STATE_NEXT_MEM[self.ending_position] = state_next.float()
        self.DONE_MEM[self.ending_position] = done.astype(float)

        self.ending_position = (
            self.ending_position + 1
        ) % self.max_memory_size  # FIFO tensor
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)

    def batch_experiences(self):
        # Randomly samples batch size experiences
        idx = np.random.choice(self.num_in_queue, size=self.memory_sample_size)
        STATE_PREV = self.STATE_PREV_MEM[idx]
        ACTION = self.ACTION_MEM[idx]
        REWARD = self.REWARD_MEM[idx]
        STATE_NEXT = self.STATE_NEXT_MEM[idx]
        DONE = self.DONE_MEM[idx]
        return STATE_PREV, ACTION, REWARD, STATE_NEXT, DONE

    def act(self, state):
        # Epsilon-greedy exploration
        # TODO figure out why it needs to be this shape
        if np.random.rand() < self.exploration_rate:
            return torch.tensor([[np.random.choice(self.action_space)]])
        else:
            action = torch.argmax(self.dqn(state.to(self.device)))
            action = action.unsqueeze(0).unsqueeze(0)
            return action

    def experience_replay(self):
        if self.memory_sample_size > self.num_in_queue:
            return

        # Sample batch of experiences
        STATE_PREV, ACTION, REWARD, STATE_NEXT, DONE = self.batch_experiences()
        STATE_PREV = STATE_PREV.to(self.device)
        ACTION = ACTION.to(self.device)
        REWARD = REWARD.to(self.device)
        STATE_NEXT = STATE_NEXT.to(self.device)
        DONE = DONE.to(self.device)

        self.optimizer.zero_grad()
        # Q-Learning target is Q*(S, A) <- r + γ max_a Q(S', a)
        target = REWARD + torch.mul(
            (self.gamma * self.dqn(STATE_NEXT).max(1).values.unsqueeze(1)), 1 - DONE
        )
        current = self.dqn(STATE_PREV).gather(1, ACTION.long())

        loss = self.l1(current, target)
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Backpropagate error

        self.exploration_rate *= self.exploration_decay

        # Makes sure that exploration rate is always at least 'exploration min'
        self.exploration_rate = max(self.exploration_rate, self.exploration_min)
