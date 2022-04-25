from typing import Union

import cv2
import numpy as np
import numpy.typing as npt

from gym import Env
from gym.spaces import Discrete, Box

from snake_game import SnakeGame

# TODOs:
# - Figure out how to handle quiting the game
# - Resolve super laggy video when using namedWindow
#   - Maybe resolve by just moving the namedWindow during the first render
# - Add a delay between selecting settings and checking for header/gamefield as it sometimes fails
#   - This should be solved but leaving here for now


class SnakeEnv(Env):
    def __init__(
        self, img_width: int = 530, img_height: int = 477, gray_scale: bool = False
    ) -> None:
        self.action_space: Discrete = Discrete(5)
        """Discrete: Environment's action space: NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]."""

        self.action_to_key: dict[int, str] = {
            0: "NOOP",
            1: "UP",
            2: "RIGHT",
            3: "DOWN",
            4: "LEFT",
        }
        """dict[int, str]: Maps actions to keys."""

        self.observation_space: Box = Box(
            low=0,
            high=255,
            shape=(img_height, img_width, 1 if gray_scale else 3),
            dtype=np.uint8,
        )
        """Box: Environment's observation space, an image of the current gamefield of shape:
            (img_height, img_width, 0 if gray_scale else 3)."""

        self.state: Union[npt.NDArray, None] = None
        """npt.NDArray or None: Current state of the environment."""

        self.game: SnakeGame = SnakeGame()
        """SnakeGame: Snake game instance."""
        self.game.calibrate()

        self.prev_score: int = 0
        """int: Previous score, helps keep track of rewards."""

        # for rendering
        self.render_window_name: str = "Snake Game"
        cv2.namedWindow(self.render_window_name, cv2.WINDOW_AUTOSIZE)

    def step(self, action):
        # Apply action to the state
        if self.action_to_key[action] != "NOOP":
            self.game.send_key(self.action_to_key[action].lower())

        # Get the new state
        self.state, score, game_over = self.game.get_state(
            img_width=self.observation_space.shape[1],
            img_height=self.observation_space.shape[0],
        )

        # calcualte the reward
        reward = score - self.prev_score
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
        return self.game.reset(
            img_width=self.observation_space.shape[1],
            img_height=self.observation_space.shape[0],
        )

    def render(self):
        # for initial case before first step
        if self.state is None:
            self._first_render()
            return

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
        cv2.imshow(self.render_window_name, img)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            self.close()

    def _first_render(self):
        """Renders the first frame of the game.

        It is neccesary to do this step seperately from the render()
        method as the window needs to be moved out of the way and then
        the focus needs to be returned to the game.
        """
        cv2.moveWindow(self.render_window_name, 2000, 1000)
        cv2.imshow(self.render_window_name, np.zeros(self.observation_space.shape))
        self.game.focus_game()

    def close(self):
        """Closes the render window and exits the game."""
        cv2.destroyAllWindows()
        # TODO: figure out how to send the signal to the train loop to keep so that the model is saved
        exit()  # for now


if __name__ == "__main__":
    import time

    print("Running a SnakeEnv test with a random agent")
    time.sleep(3)
    env = SnakeEnv(img_width=10, img_height=9)

    episode_count = 10
    for episode in range(episode_count):
        obs = env.reset()
        done = False
        tot_score = 0

        prev_time = time.time()
        avg_fps = 0
        frame_count = 0
        while not done:
            env.render()
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            tot_score += reward

            # Counting FPS
            fps = 1 / (time.time() - prev_time)
            prev_time = time.time()
            avg_fps = (avg_fps * frame_count + fps) / (frame_count + 1)

        print(f"Episode {episode} finished with score {tot_score}")
        print(f"Average FPS: {avg_fps}")

