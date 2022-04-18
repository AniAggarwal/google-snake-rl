import cv2
import numpy as np
import numpy.typing as npt

from gym import Env
from gym.spaces import Discrete, Box

from snake_game import SnakeGame


class SnakeEnv(Env):
    def __init__(
        self, img_width: int = 530, img_height: int = 477, gray_scale: bool = False
    ) -> None:
        self.action_space = Discrete(5)
        """gym.spaces.Discrete: Environment's action space: NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]."""

        self.action_to_key = {0: "NOOP", 1: "UP", 2: "RIGHT", 3: "DOWN", 4: "LEFT"}
        """dict[int, str]: Maps actions to keys."""

        self.observation_space = Box(
            low=0,
            high=255,
            shape=(img_height, img_width, 1 if gray_scale else 3),
            dtype=np.uint8,
        )
        """gym.spaces.Box: Environment's observation space, an image of the current gamefield of shape:
            (img_height, img_width, 0 if gray_scale else 3)."""

        self.state = None
        """npt.NDArray or None: Current state of the environment."""

        self.game = SnakeGame()
        """snake_game.SnakeGame: Snake game instance."""
        self.game.calibrate(monitor_num=3)

        self.prev_score = 0
        """int: Previous score, helps keep track of rewards."""

        # for rendering
        self.render_window = cv2.namedWindow("Snake Game", cv2.WND_PROP_TOPMOST)

    def step(self, action):
        # Apply action to the state
        if self.action_to_key[action] != "NOOP":
            self.game.send_key(self.action_to_key[action].lower())

        # Get the new state
        self.state, score, game_over = self.game.get_state()

        # calcualte the reward
        reward = self.prev_score - score
        self.prev_score = score
        # TODO figure out if I should punish for losing the game
        if game_over:
            self.prev_score = 0

        # no info to return for now
        info = {}

        # Return the new state, reward, whether the game is over, and info
        return self.state, reward, game_over, info

    def reset(self):
        # the first move cannot be NOOP as the game won't start
        # TODO figure out how to resolve this
        self.state = None
        self.prev_score = 0
        self.game.reset()

    def render(self):
        img = self.state.copy()
        cv2.putText(
            img,
            f"Score: {self.prev_score}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.moveWindow(self.render_window, 1000, 100)
        cv2.imshow(self.render_window, img)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            cv2.destroyAllWindows()
            # TODO figure out if I should just kill this program here
            exit()


if __name__ == "__main__":
    env = SnakeEnv()
    env.reset()
    for i in range(1000):
        env.step(env.action_space.sample())
        env.render()
