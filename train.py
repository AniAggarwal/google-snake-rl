import time
import pickle

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from snake_env import SnakeEnv
from dqn_model import DQNAgent

# Code from https://gist.github.com/Jawdeyu/1d633c35238d13484deb2969ff40005d#file-dqn_run-py


def run(training_mode, pretrained, num_episodes=1000, exploration_max=1):
    print("Running SnakeEnv training...")
    time.sleep(3)
    env = SnakeEnv(img_width=10, img_height=9)

    observation_space = env.observation_space.shape
    action_space = env.action_space.n
    agent = DQNAgent(
        state_space=observation_space,
        action_space=action_space,
        max_memory_size=30000,
        batch_size=32,
        gamma=0.90,
        lr=0.00025,
        dropout=0.2,
        exploration_max=exploration_max,
        exploration_min=0.02,
        exploration_decay=0.99,
        pretrained=pretrained,
    )

    # Restart the enviroment for each episode
    num_episodes = num_episodes
    env.reset()

    total_rewards = []
    if training_mode and pretrained:
        with open("total_rewards.pkl", "rb") as f:
            total_rewards = pickle.load(f)

    for ep_num in tqdm(range(num_episodes)):
        state = torch.Tensor([env.reset()])
        done = False
        total_reward = 0

        while not done:
            # TODO add flag in function for rendering
            # env.render()

            action = agent.act(state)

            state_next, reward, done, info = env.step(int(action[0]))
            total_reward += reward

            state_next = torch.Tensor([state_next])
            reward = torch.tensor([reward]).unsqueeze(0)

            if training_mode:
                # if using done as a bool causes issues, switch to int
                agent.remember(state, action, reward, state_next, done)
                # TODO probably do experience_replay after game is over instead of after every action
                # This is bc game does not pause between steps
                agent.experience_replay()

            state = state_next

        total_rewards.append(total_reward)

        if ep_num != 0 and ep_num % 100 == 0:
            print(
                "Episode {} score = {}, average score = {}".format(
                    ep_num + 1, total_rewards[-1], np.mean(total_rewards)
                )
            )
        num_episodes += 1

    # Save the trained memory so that we can continue from where we stop using 'pretrained' = True
    if training_mode:
        with open("ending_position.pkl", "wb") as f:
            pickle.dump(agent.ending_position, f)
        with open("num_in_queue.pkl", "wb") as f:
            pickle.dump(agent.num_in_queue, f)
        with open("total_rewards.pkl", "wb") as f:
            pickle.dump(total_rewards, f)

        torch.save(agent.dqn.state_dict(), "DQN.pt")
        torch.save(agent.STATE_PREV_MEM, "STATE_PREV_MEM.pt")
        torch.save(agent.ACTION_MEM, "ACTION_MEM.pt")
        torch.save(agent.REWARD_MEM, "REWARD_MEM.pt")
        torch.save(agent.STATE_NEXT_MEM, "STATE_NEXT_MEM.pt")
        torch.save(agent.DONE_MEM, "DONE_MEM.pt")

    env.close()


if __name__ == "__main__":
    run(num_episodes=1000, training_mode=True, pretrained=False)
