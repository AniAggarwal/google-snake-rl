# Snake Game Environment README

This is a reinforcement learning environment based on the classic Snake game, implemented using OpenAI Gym. This environment allows you to train and evaluate agents that can learn to play Snake game.

## Installation

The environment is built with the following dependencies:

-   OpenCV
-   NumPy
-   Gym

You can install these dependencies using the following command:

`pip install opencv-python numpy gym` 

## Usage

The environment can be used just like any other OpenAI Gym environment. Here is an example:

```python
import gym

env = gym.make('snake_game:SnakeEnv')
observation = env.reset()

for t in range(100):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    env.render()

    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
env.close()
```

## Parameters

The environment constructor takes in the following parameters:

-   `img_width`: The width of the game screen (default=530).
-   `img_height`: The height of the game screen (default=477).
-   `gray_scale`: Whether to use grayscale image for the observation space (default=False).

## Actions

The environment has 5 possible actions:

-   `0`: NOOP (no operation)
-   `1`: UP
-   `2`: RIGHT
-   `3`: DOWN
-   `4`: LEFT

## Observation Space

The observation space is an image of the current game field. The shape of the observation space is (img_height, img_width, 3) if `gray_scale` is False and (img_height, img_width, 1) if `gray_scale` is True. The values in the observation space range from 0 to 255.

## Methods

The environment has the following methods:

-   `step(action)`: Execute an action in the environment and return the observation, reward, done, and info.
-   `reset()`: Reset the environment to its initial state and return the observation.
-   `render()`: Render the current state of the environment.
-   `close()`: Close the environment and release all resources.

## Note

The current implementation of the environment does not handle quitting the game properly. If you want to quit the game, you need to close the window manually.

Also, the video rendering may be laggy when using the named window. The window may need to be moved out of the way in the first render to avoid this.

## License

This code is licensed under the GPL License.
