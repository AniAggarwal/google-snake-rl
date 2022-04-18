from gym import Env
from gym.spaces import Discrete, Box


class SnakeEnv(Env):
    def __init__(
        self, img_width: int = 530, img_height: int = 477, gray_scale: bool = False
    ) -> None:
        self.action_space = Discrete(5)
        """gym.spaces.Discrete: Environment's action space: NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]."""
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(img_height, img_width, 1 if gray_scale else 3),
            dtype=np.uint8,
        )
        """gym.spaces.Box: Environment's observation space, an image of the current gamefield of shape:
            (img_height, img_width, 0 if gray_scale else 3)."""

        self.state = None

        self.game = SnakeGame()

    def step(self, action):
        # Apply action to the state

        pass

    def render(self):
        pass

    def reset(self):
        pass
