import time
from collections import deque
from pathlib import Path
from pathlib import PurePath
import warnings
from typing import Union, Iterable, List, Tuple, Dict

import numpy as np
import numpy.typing as npt
import cv2

from mss.linux import MSS as mss
import pyautogui

# List of TODOs:
# - Sort out the verbose mode


class SnakeGame:
    """An interface to retrive information from the Google Snake Game.

    Link to the game: https://www.google.com/fbx?fbx=snake_arcade

    Parameters
    ----------
    monitor_num : int
        The number of the monitor to use. Defaults to None, and if not
        specified during instantiation or in calibrate(), the 'last'
        monitor will be used.
    template_path : str or PurePath
        The path to the directory containing the templates. Defaults to
        None, which will use the templates in the SnakeGame directory.
    verbose_mode : bool
        Whether to print out extra information. Defaults to False.

    Methods
    -------
    calibrate()
        Calibrates the game settings and bounding boxes.
    get_state()
        Gets the current game state in the form of an image of the gamefield.
    reset()
        Resets the game.
    send_key(key)
        Sends a key to the game.
    focus_game()
        Focuses on the game by clicking on the gamefield area.
    """

    def __init__(
        self,
        monitor_num: int = None,
        template_path: Union[str, PurePath] = None,
        verbose_mode: bool = False,
    ):
        self._monitor_num = monitor_num
        """int: The number of the monitor to use."""
        self._verbose_mode = verbose_mode
        """bool: Whether to print out extra information."""

        # self._thresh_header = np.array([[48, 159, 100], [48, 159, 125]])
        self._thresh_header = np.array([[51, 150, 0], [80, 180, 255]])
        """npt.NDArray: The color threshold for the header."""
        # self._thresh_gamefield = np.array([[40, 150, 0], [40, 180, 255]])
        self._thresh_gamefield = np.array([[45, 0, 0], [50, 200, 255]])
        """npt.NDArray: The color threshold for the gamefield."""

        if template_path is None:
            template_path = Path("./templates").resolve()
        if not isinstance(template_path, PurePath):
            template_path = Path(template_path).resolve()
        self._template_path = template_path
        """Path: The path to the templates to use for matches."""
        self._icon_template_path = self._template_path / "icons"
        """Path: The path to the icons templates to use for UI element matches."""
        self._digit_template_path = self._template_path / "digits"
        """Path: The path to the digits templates to use to find the score."""

        self._settings_pixel_mapping = {
            "apple": (151, 65),
            "wall": (198, 109),
            "5_balls": (242, 157),
            "snake": (148, 205),
            "small_field": (198, 241),
            "blue_snake": (142, 287),
            "play_button": (118, 374),
        }
        """dict[str, tuple]: Mappings of number of pixels away from
        top left of settings-page.png to each setting."""
        self._header_pixel_bboxes = {
            "digit_start_top_left": (59, 29),
            "digit_end_bottom_right": (125, 47),
        }
        """dict[str, tuple]: Mapping of bboxes of the area numbers
        appear in the header relative to top left of header bbox."""

        self._sct = mss()
        """mss: The mss object to use for taking screenshots."""

    def calibrate(self, monitor_num: int = None, try_count : int = 3) -> None:
        """Calibrates the game.

        Calibrates by settings up the correct settings and finding
        the bounding boxes of the game elements.

        Parameters
        ----------
        monitor_num : int
            The number of the monitor to use. Defaults to None, which
            will use the monitor number specified in the SnakeGame object.
            If not specified here or in the SnakeGame object, will use
            the 'last' monitor.
        try_count : int
            The number of times to try calibrating. Defaults to 3.

        Returns:
        --------
        None

        Raises
        ------
        ValueError if the game cannot be calibrated.

        Warnings
        --------
        If the monitor number is not specified and the SnakeGame
        object does not have a monitor number specified.
        If the gamefield or header bounding box is not found.
        """

        if monitor_num is not None:
            self._monitor_num = monitor_num
            """int: The number of the monitor to use."""
        if self._monitor_num is None:
            self._monitor_num = len(mss().monitors) - 1 
            warnings.warn(f"No monitor number specified, using {self._monitor_num}.")

        # will set game settings and get game bounding box
        self._select_game_settings(match_thresh=0.95)
        time.sleep(1)  # wait for game to load

        # MUST BE CALLED BEFORE PRESSING PLAY
        self._header_bbox = None
        self._gamefield_bbox = None
        for attempt_num in range(try_count + 1):
            # if last attempt failed use manual calibration
            if attempt_num == try_count:
                input("Using manual calibration. Please press play and ensure the"
                      f"gamefield and header is visible on monitor number {self._monitor_num}."
                      "Press enter once ready.")
                time.sleep(1)
                with mss() as sct:
                    img = np.array(sct.grab(sct.monitors[self._monitor_num]))
                self._thresh_header = self._manual_hsv_calibration(img, self._thresh_header)
                self._thresh_gamefield = self._manual_hsv_calibration(img, self._thresh_gamefield)
                time.sleep(1)
            try:
                self._header_bbox, self._gamefield_bbox = self._get_game_bboxes()
            except ValueError as _:
                if attempt_num == try_count:
                    # TODO bug here
                    raise ValueError("Could not find gamefield or header bounding box.")
                else:
                    warnings.warn(f"Could not find gamefield and/or header. Attempt {attempt_num + 1}.")
                    time.sleep(1)

        if self._header_bbox is None or self._gamefield_bbox is None:
            input("Using manual calibration. Please press play and ensure the"
                  f"gamefield and header is visible on monitor number {self._monitor_num}."
                  "Press enter once ready.")
            time.sleep(1)
            with mss() as sct:
                img = np.array(sct.grab(sct.monitors[self._monitor_num]))
            self._thresh_header = self._manual_hsv_calibration(img, self._thresh_header)
            self._thresh_gamefield = self._manual_hsv_calibration(img, self._thresh_gamefield)

        # keep history of last few scores in case of screenshot glitches
        self._score_deque = deque(maxlen=3)
        # the game always starts with a score of 0 and adding this prevents index errors
        self._score_deque.append(0)

    def get_state(self, img_width: int = None, img_height: int = None) -> Tuple[npt.NDArray, int, bool]:
        """Gets the current game state as an image of the gamefield.

        Parameters
        ----------
        img_width : int
            The width of the image to return. Defaults to None, which
            will use the default width of the gamefield.
        img_height : int
            The height of the image to return. Defaults to None, which
            will use the default height of the gamefield.

        Returns
        -------
        img : npt.NDArray
            The image of the gamefield of shape (477, 530, 3). 
        score : int
            The current score or the last score if the game has ended.
        game_over : bool
            True if the game is over, False otherwise.

        Raises
        ------
        ValueError
            If img_width or img_height is provided but not both.
        """
        header = np.array(self._sct.grab(self._header_bbox))
        gamefield = np.array(self._sct.grab(self._gamefield_bbox))

        # check if game is over
        # TODO add a way to pass in a header thresh here
        game_over = self._check_game_end(header)
        if game_over:
            # use the last score instead of calling _get_header_score()
            # because we cannot see the score properly after the game ends
            score = self._score_deque[-1]
        else:
            # get the current score
            score = self._get_header_score(header)
            if score is not None:
                self._score_deque.append(score)
            else:
                score = self._score_deque[-1]

        if self._verbose_mode:
            cv2.putText(
                gamefield,
                f"Score: {score}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.imshow("game", gamefield)

        # mss is of format BGRA and we only want BGR
        gamefield = gamefield[:, :, :3]
        # resize to fit the desired image size
        if (img_width is None) ^ (img_height is None):
            raise ValueError("Must specify both img_width and img_height or neither.")
        if img_width is not None and img_height is not None:
            gamefield = cv2.resize(gamefield, (img_width, img_height))
        return gamefield, score, game_over

    def reset(self) -> None:
        """Resets the game by pressing the play button.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # first focus on game by clicking on the gamefield area
        self.focus_game()
        pyautogui.press("enter")
        time.sleep(0.5)

    def send_key(self, key: str) -> None:
        """Sends a key to the game.

        Parameters
        ----------
        key : str
            The key to send, must be either 'up', 'down', 'left', or 'right'.

        Returns
        -------
        None

        Raises
        ------
        ValueError if the key is not a valid key.
        """
        if key not in ["up", "down", "left", "right"]:
            raise ValueError(
                f"Key {key} not recognized. Must be one of 'up', 'down', 'left', or 'right'."
            )
        pyautogui.press(key)

    def focus_game(self) -> None:
        """Focuses on the game by clicking on the gamefield area.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # first focus on game by clicking on the gamefield area
        pyautogui.click(x=self._gamefield_bbox["left"], y=self._gamefield_bbox["top"])

    def _check_game_end(
        self, header_img: npt.NDArray, header_thresh: float = 30.0
    ) -> bool:
        """Checks if the game is over.

        Parameters
        ----------
        header_img : npt.NDArray
            The image of the header to check if the game is over.
        header_thresh : float
            The threshold to use for the game end check.
            Defaults to 30.0. Calibrate this by looking
            at the mean header while game is and isn't over.

        Returns
        -------
        bool
            True if the game is over, False otherwise.
        """
        # TODO add a way to manually calibrate this
        # check if game is over by seeing if there has been a signficant change in the mean header img
        return np.mean(header_img) < header_thresh

    def _get_header_score(self, header_img: npt.NDArray) -> Union[int, None]:
        """Gets the score from an image of the header.

        Will not look at the high score, only the current score.

        Parameters
        ----------
        img : npt.NDArray
            The image of the header to get the score from.

        Returns
        -------
        number : Union[int, None]
            The current score, returns None if no score is found.
        """
        header_img = cv2.cvtColor(header_img, cv2.COLOR_BGR2GRAY)
        _, header_img = cv2.threshold(
            header_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )

        score_roi = header_img[
            self._header_pixel_bboxes["digit_start_top_left"][
                1
            ] : self._header_pixel_bboxes["digit_end_bottom_right"][1],
            self._header_pixel_bboxes["digit_start_top_left"][
                0
            ] : self._header_pixel_bboxes["digit_end_bottom_right"][0],
        ]

        # check if the score roi is empty if no socre or white due to glitched screenshot
        if np.count_nonzero(score_roi) == 0 or np.count_nonzero(score_roi == 0) == 1:
            return None

        # using a range as DIGIT_TEMPLATE_PATH.iterdir() returns a random order
        template_imgs = [
            cv2.imread(str(self._digit_template_path / f"{digit}.png"), 0)
            for digit in range(10)
        ]

        # use vertical lines to see if we are at the end of a digit
        # flatten score_roi to 1D array where areas where there is part of a
        # digit vertically are 1, rest are 0. Will be a bool 1D array
        score_roi_flat = np.any(score_roi, axis=0)

        if self._verbose_mode:
            score_roi_disp = score_roi_flat.copy().astype(np.uint8).reshape(-1, 1)
            score_roi_disp = np.repeat(score_roi_disp, 18, 1).T * 255
            cv2.imshow("flat roi", score_roi_disp)
            cv2.imshow("score roi", score_roi)

        # find the start and end of each digit
        digit_start = None  # inclusive
        digit_end = None  # exclusive

        # a list of all the digits found
        digits = []
        for x_pos, is_digit in enumerate(score_roi_flat):
            if x_pos < score_roi_flat.shape[0] - 1:
                # if next pixel is start of digit
                if not is_digit and score_roi_flat[x_pos + 1]:
                    digit_start = x_pos + 1
                # else if current pixel is end of digit
                elif is_digit and not score_roi_flat[x_pos + 1]:
                    digit_end = x_pos + 1

            # check if we have found a digit
            if digit_start is not None and digit_end is not None:
                digit_img = score_roi[:, digit_start:digit_end]
                # reset start and end
                digit_start, digit_end = None, None

                # find the digit with the highest match
                ious = []
                for template_img in template_imgs:
                    if template_img.shape[1] < digit_img.shape[1]:
                        pad_width = [
                            [0, 0],
                            [0, digit_img.shape[1] - template_img.shape[1]],
                        ]
                        template_img = np.pad(template_img, pad_width, "constant")
                    elif template_img.shape[1] > digit_img.shape[1]:
                        pad_width = [
                            [0, 0],
                            [0, template_img.shape[1] - digit_img.shape[1]],
                        ]
                        digit_img = np.pad(digit_img, pad_width, "constant")

                    ious.append(
                        np.count_nonzero(np.logical_and(digit_img, template_img))
                        / np.count_nonzero(np.logical_or(digit_img, template_img))
                    )
                best_digit = max(range(len(ious)), key=ious.__getitem__)
                digits.append(best_digit)

        return (
            None if len(digits) == 0 else int("".join([str(digit) for digit in digits]))
        )

    def _select_game_settings(self, match_thresh: float = 0.9) -> None:
        """Selects the game settings.

        Parameters
        ----------
        match_thresh : float
            How tolerate to be for imperfect matches.

        Returns
        -------
        None

        Warnings
        --------
        If settings are not found or already selected.
        """
        # TODO add a way to pass two different match threshs
        # find and click settings icon
        settings_center = self._find_template_match(
            self._icon_template_path / "settings-icon.png",
            match_thresh=match_thresh,
            get_center=True,
        )
        if settings_center is None:
            warnings.warn("Could not find settings icon. Skipping settings selection.")
            return
        pyautogui.click(x=settings_center[0], y=settings_center[1])
        # wait for settings to load
        time.sleep(0.5)

        # use absolute positioning relative to settings page template match
        # to select all settings (because template match wasn't working great)
        settings_page_corner = self._find_template_match(
            self._icon_template_path / "settings-page.png",
            get_center=False,
            match_thresh=match_thresh,
        )
        if settings_page_corner is None:
            warnings.warn(
                "Could not find settings page. Skipping settings selection and starting game."
            )
            pyautogui.press("enter")
            return

        for _, point in self._settings_pixel_mapping.items():
            pyautogui.click(
                x=point[0] + settings_page_corner[0],
                y=point[1] + settings_page_corner[1],
            )
            time.sleep(0.1)

    def _find_template_match(
        self,
        template_path: Union[str, PurePath],
        match_thresh: float = 0.9,
        get_center: bool = False,
    ) -> Union[Tuple[int, int], None]:
        """Finds the template match.

        Parameters
        ----------
        template_path : Union[str, PurePath]
            Path to the template.
        match_thresh : float
            How tolerate to be for imperfect matches.
        get_center : bool
            Whether to return the center of the template match or
            the top left corner.

        Returns
        -------
        Union[Tuple[int, int], None]
            The center of the template match if get_center is True, otherwise
            the top left corner of the template match.
        """
        with mss() as sct:
            # screenshot is in BGRA format, but we need BGR
            img = np.array(sct.grab(sct.monitors[self._monitor_num]))[:, :, :3]

        template = cv2.imread(str(template_path))

        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

        # check to make sure a match was found
        # done by making sure the best match point is above the match threshold
        if np.amax(res) < match_thresh:
            if self._verbose_mode:
                cv2.imshow("template", template)
                cv2.imshow("img", img)
                cv2.waitKey(-1)
            return None

        max_loc = cv2.minMaxLoc(res)[-1]
        top_left = max_loc
        bottom_right = (
            top_left[0] + template.shape[1],
            top_left[1] + template.shape[0],
        )

        if get_center:
            match_point = (
                top_left[0] + template.shape[1] // 2,
                top_left[1] + template.shape[0] // 2,
            )
            match_point = self._relative_point_to_abs(match_point)
        else:
            match_point = self._relative_point_to_abs(top_left)

        if self._verbose_mode:
            print(top_left, bottom_right)
            print(
                f"x width: {bottom_right[0] - top_left[0]}, y height: {bottom_right[1] - top_left[1]}"
            )
            print(f"center: {match_point}")
            img = img.copy()  # Needed so that cv2.rectangle doesn't throw an error
            cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)
            cv2.imshow("template match", img)
            cv2.waitKey(-1)

        return match_point

    def _relative_point_to_abs(self, point: Tuple[int, int]) -> Tuple[int, int]:
        """Converts a point relative to the monitor to an absolute point.

        Parameters
        ----------
        point : Tuple[int, int]
            The point relative to the monitor.

        Returns
        -------
        point : Tuple[int, int]
            The absolute point.
        """
        with mss() as sct:
            monitor = sct.monitors[self._monitor_num]
            return (point[0] + monitor["left"], point[1] + monitor["top"])

    def _get_game_bboxes(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Gets bounding boxes for the gamefield and header.

        Returns
        -------
        header_bbox : dict
            The absolute bounding box for the header.
        gamefield_bbox : dict
            The absolute bounding box for the gamefield.
        """
        with mss() as sct:
            img = np.array(sct.grab(sct.monitors[self._monitor_num]))

        if self._verbose_mode:
            cv2.imshow("full frame", img)
            cv2.waitKey(-1)

        header_bbox = self._get_bbox_by_hsv(img, self._thresh_header)
        gamefield_bbox = self._get_bbox_by_hsv(img, self._thresh_gamefield)
        header_bbox = self._relative_bbox_to_abs(header_bbox)
        gamefield_bbox = self._relative_bbox_to_abs(gamefield_bbox)
        return header_bbox, gamefield_bbox

    def _get_bbox_by_hsv(self, img: npt.NDArray, thresh: npt.NDArray) -> Dict[str, int]:
        """Gets bounding box by provided HSV thresholding.

        Will use the given image to find the single best rectangle that
        fits the threshold color.

        Parameters
        ----------
        img : npt.NDArray
            The image to look for the bounding box in.
        thresh : npt.NDArray
            A numpy array of the form
            [[lower_h, lower_s, lower_v], [upper_h, upper_s, upper_v]].

        Returns
        -------
        bbox : dict[str, int]
            The box bounding the given color in form
            {'left': int, 'top': int, 'width': int, 'height': int}.

        Raises
        ------
        ValueError
            If the bounding box is not found using given HSV thresholding.
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(img, thresh[0, :], thresh[1, :])
        masked = cv2.cvtColor(cv2.bitwise_and(img, img, mask=mask), cv2.COLOR_HSV2BGR)
        masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(masked, 0, 255, cv2.THRESH_OTSU)

        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        if self._verbose_mode:
            img = cv2.cvtColor(img.copy(), cv2.COLOR_HSV2BGR)
            cv2.imshow("masked", masked)
            cv2.imshow("thresh", thresh)
            cv2.waitKey(-1)

        if len(contours) == 0:
            raise ValueError("Could not find gamefield and/or header.")

        rects = []
        for contour in contours:
            # only keep the contours that are nearly perfect rectangles and are large enough
            # holes in the trophy are detected as rectangles so area is filtered too
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.001 * perimeter, True)
            area_percentage = cv2.contourArea(contour) / (img.shape[0] * img.shape[1])
            if len(approx) == 4 and area_percentage > 0.01:
                rects.append(contour)

        if self._verbose_mode:
            cv2.drawContours(img, rects, -1, (0, 255, 0), 3)
            cv2.imshow("bbox", img)
            cv2.waitKey(-1)

        if len(rects) != 1:
            raise ValueError("Could not find gamefield and/or header.")

        return {
            label: val
            for label, val in zip(
                ["left", "top", "width", "height"], cv2.boundingRect(np.array(rects[0]))
            )
        }

    def _relative_bbox_to_abs(self, bbox: Dict[str, int]) -> Dict[str, int]:
        """Converts a bounding box relative to the monitor to an absolute bounding box.

        Parameters
        ----------
        bbox : dict
            The bounding box relative to the monitor.

        Returns
        -------
        bbox : dict
            The absolute bounding box.
        """
        with mss() as sct:
            monitor = sct.monitors[self._monitor_num]
            bbox["left"] += monitor["left"]
            bbox["top"] += monitor["top"]
            return bbox

    def _manual_hsv_calibration(self, img : npt.NDArray, default_thresh : npt.NDArray = np.array([[0, 0, 0], [179, 255, 255]])) -> npt.NDArray:
        """Manually calibrate the HSV thresholding.

        This will show the image and allow the user to select the HSV
        thresholding values.

        Parameters
        ----------
        img : npt.NDArray
            The image to use for calibration.
        default_thresh : npt.NDArray
            The default HSV thresholding values in the form of a numpy
            array: [[lower_h, lower_s, lower_v], [upper_h, upper_s, upper_v]].

        Returns
        -------
        thresh : npt.NDArray
            The HSV thresholding values in the form of a numpy array:
            [[lower_h, lower_s, lower_v], [upper_h, upper_s, upper_v]].
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        win_name = "Calibration. Press 'enter' when complete."
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        # Create trackbars for color change
        # Hue is from 0-179 for Opencv
        cv2.createTrackbar('H_min', win_name, 0, 179, lambda x: None)
        cv2.createTrackbar('S_min', win_name, 0, 255, lambda x: None)
        cv2.createTrackbar('V_min', win_name, 0, 255, lambda x: None)
        cv2.createTrackbar('H_max', win_name, 0, 179, lambda x: None)
        cv2.createTrackbar('S_max', win_name, 0, 255, lambda x: None)
        cv2.createTrackbar('V_max', win_name, 0, 255, lambda x: None)

        # Set default value for the trackbars
        cv2.setTrackbarPos('H_min', win_name, default_thresh[0, 0])
        cv2.setTrackbarPos('S_min', win_name, default_thresh[0, 1])
        cv2.setTrackbarPos('V_min', win_name, default_thresh[0, 2])
        cv2.setTrackbarPos('H_max', win_name, default_thresh[1, 0])
        cv2.setTrackbarPos('S_max', win_name, default_thresh[1, 1])
        cv2.setTrackbarPos('V_max', win_name, default_thresh[1, 2])

        # Initialize HSV min/max values
        h_min = s_min = v_min = h_max = s_max = v_max = 0

        while True:
            # Get current positions of all trackbars
            h_min = cv2.getTrackbarPos('H_min', win_name)
            s_min = cv2.getTrackbarPos('S_min', win_name)
            v_min = cv2.getTrackbarPos('V_min', win_name)
            h_max = cv2.getTrackbarPos('H_max', win_name)
            s_max = cv2.getTrackbarPos('S_max', win_name)
            v_max = cv2.getTrackbarPos('V_max', win_name)

            # Set minimum and maximum HSV values to display
            lower_hsv = np.array([h_min, s_min, v_min])
            upper_hsv = np.array([h_max, s_max, v_max])

            # Mask and display img
            mask = cv2.inRange(img, lower_hsv, upper_hsv)
            res = cv2.bitwise_and(img, img, mask=mask)
            cv2.imshow(win_name, res)

            # Wait for escape key to exit
            # if cv2.waitKey(1) == 27:
            if cv2.waitKey(1) & 0xFF == ord('\r'):
                break
                cv2.destroyAllWindows()

        print(f"Using manual calibration values: {np.array([[h_min, s_min, v_min], [h_max, s_max, v_max]])}")
        return np.array([[h_min, s_min, v_min], [h_max, s_max, v_max]])


if __name__ == "__main__":
    game = SnakeGame(3)
    game.calibrate()
    while True:
        gamefield, score, game_over = game.get_state()

        if game_over:
            print("Game over")
            pyautogui.press("enter")
            time.sleep(1)
        else:
            img = gamefield.copy()
            cv2.putText(
                img,
                f"Score: {score}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.imshow("gamefield", img)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            cv2.destroyAllWindows()
            break

