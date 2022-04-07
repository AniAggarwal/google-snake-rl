import time
from collections import deque
from pathlib import Path
from pathlib import PurePath
from typing import Union, Iterable, List, Tuple, Dict

import numpy as np
import numpy.typing as npt
import cv2

from mss.linux import MSS as mss
import pyautogui

# Game url:
# https://www.google.com/fbx?fbx=snake_arcade

# TODO list:
# - there are some weird glitches where screenshot is completely white in certain parts
#   - fix this by keeping 3-5 frame average of score
# - detect perfect template matches
# - detect game end
# - docstrings for all working functions
# - create a neatly packaged class of all this later
# - if settings icon is found but no settings page, assume settings are already set and press play

THRESH_HEADER = np.array([[48, 159, 100], [48, 159, 125]])
THRESH_GAMEFIELD = np.array([[40, 150, 0], [40, 180, 255]])

TEMPLATE_PATH = Path("./templates").resolve()
ICON_TEMPLATE_PATH = TEMPLATE_PATH / "icons"
DIGIT_TEMPLATE_PATH = TEMPLATE_PATH / "digits"

# Mappings of number of pixels away from top left of settings-page.png to each setting
SETTINGS_PIXEL_MAPPING = {
    "apple": (151, 65),
    "wall": (198, 109),
    "5_balls": (242, 157),
    "snake": (148, 205),
    "small_field": (198, 253),
    "blue_snake": (142, 303),
    "play_button": (118, 374),
}
# Mapping of bboxes of the area numbers appear in the header relative to top left of header bbox
HEADER_PIXEL_BBOXES = {
    # "full_bbox": [[55, 28], [120, 49]],
    # "top_score": [[161, 28], [226, 49]],
    "digit_start_top_left": [59, 29],
    "digit_end_bottom_right": [125, 47],
    "digit_width": 11,
    "digit_height": 18,
}


def get_bbox_by_hsv(
    img: npt.NDArray, thresh: npt.NDArray, display_bbox: bool = False
) -> Dict[str, int]:
    """
    Gets bounding box by provided HSV thresholding.

    Will use the given image to find the single best rectangle that fits the threshold color.

    Parameters
    ----------
    img : npt.NDArray
        The image to look for the bounding box in.
    thresh : npt.NDArray
        A numpy array of the form [[lower_h, lower_s, lower_v], [upper_h, upper_s, upper_v]].
    display_bbox : bool
        Whether to display the bounding box at each step. Defaults to False.

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

    if display_bbox:
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

    if display_bbox:
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


def get_game_bboxes(
    monitor_num: int = 0, display_bbox: bool = False
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Gets bounding boxes for the gamefield and header.

    Parameters
    ----------
    monitor_num : int
        The monitor number to use. Defaults to 0, which is all monitors stitched together.
    display_bbox : bool
        Whether to display the bounding boxes at each step. Defaults to False.

    Returns
    -------
    header_bbox : dict
        The absolute bounding box for the header.
    gamefield_bbox : dict
        The absolute bounding box for the gamefield.
    """
    with mss() as sct:
        img = np.array(sct.grab(sct.monitors[monitor_num]))

    if display_bbox:
        cv2.imshow("full frame", img)
        cv2.waitKey(-1)

    header_bbox = get_bbox_by_hsv(img, THRESH_HEADER, display_bbox)
    gamefield_bbox = get_bbox_by_hsv(img, THRESH_GAMEFIELD, display_bbox)
    header_bbox = relative_bbox_to_abs(header_bbox, monitor_num)
    gamefield_bbox = relative_bbox_to_abs(gamefield_bbox, monitor_num)
    return header_bbox, gamefield_bbox


def relative_bbox_to_abs(bbox: Dict[str, int], monitor_num: int = 0) -> Dict[str, int]:
    """
    Converts a bounding box relative to the monitor to an absolute bounding box.

    Parameters
    ----------
    bbox : dict
        The bounding box relative to the monitor.
    monitor_num : int
        The monitor number to use. Defaults to 0, which is all monitors stitched together.

    Returns
    -------
    bbox : dict
        The absolute bounding box.
    """
    with mss() as sct:
        monitor = sct.monitors[monitor_num]
        bbox["left"] += monitor["left"]
        bbox["top"] += monitor["top"]
        return bbox


def relative_point_to_abs(
    point: Tuple[int, int], monitor_num: int = 0
) -> Tuple[int, int]:
    """
    Converts a point relative to the monitor to an absolute point.

    Parameters
    ----------
    point : Tuple[int, int]
        The point relative to the monitor.
    monitor_num : int
        The monitor number to use. Defaults to 0, which is all monitors stitched together.

    Returns
    -------
    point : Tuple[int, int]
        The absolute point.
    """
    with mss() as sct:
        monitor = sct.monitors[monitor_num]
        return (point[0] + monitor["left"], point[1] + monitor["top"])


def find_template_match(
    template_path: Union[str, PurePath],
    match_thresh: float = 0.9,
    get_center: bool = False,
    monitor_num: int = 0,
    display_bbox: bool = False,
) -> Union[Tuple[int, int], None]:
    with mss() as sct:
        # screenshot is in BGRA format, but we need BGR
        img = np.array(sct.grab(sct.monitors[monitor_num]))[:, :, :3]

    template = cv2.imread(str(template_path))

    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

    # check to make sure a match was found
    # done by making sure the best match point is above the match threshold
    if np.amax(res) < match_thresh:
        return None

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])

    if get_center:
        match_point = (
            top_left[0] + template.shape[1] // 2,
            top_left[1] + template.shape[0] // 2,
        )
        match_point = relative_point_to_abs(match_point, monitor_num)
    else:
        match_point = relative_point_to_abs(top_left, monitor_num)

    if display_bbox:
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


def select_game_settings(
    match_thresh: float = 0.9, monitor_num: int = 0, display_bbox: bool = False
) -> None:
    """
    Selects the game settings.

    Parameters
    ----------
    monitor_num : int
        The monitor number to use. Defaults to 0, which is all monitors stitched together.
    display_bbox : bool
        Whether to display the bounding boxes at each step. Defaults to False.

    Returns
    -------
    None
    """
    # find and click settings icon
    settings_center = find_template_match(
        ICON_TEMPLATE_PATH / "settings-icon.png",
        match_thresh=match_thresh,
        get_center=True,
        monitor_num=monitor_num,
        display_bbox=display_bbox,
    )
    if settings_center is None:
        print("Could not find settings icon. Skipping settings selection.")
        return
    pyautogui.click(x=settings_center[0], y=settings_center[1])
    # wait for settings to load
    time.sleep(0.5)

    # use absolute positioning relative to settings page template match
    # to select all settings (because template match wasn't working great)
    settings_page_corner = find_template_match(
        ICON_TEMPLATE_PATH / "settings-page.png",
        get_center=False,
        match_thresh=match_thresh,
        monitor_num=monitor_num,
        display_bbox=display_bbox,
    )
    if settings_page_corner is None:
        print(
            "Could not find settings page. Skipping settings selection and starting game."
        )
        pyautogui.press("enter")
        return

    for _, point in SETTINGS_PIXEL_MAPPING.items():
        pyautogui.click(
            x=point[0] + settings_page_corner[0], y=point[1] + settings_page_corner[1]
        )
        time.sleep(0.1)


# def split_img_to_digits(number_img: npt.NDArray) -> List[int]:
#     """
#     Splits a pre-processed header image into each of its digits.

#     Parameters
#     ----------
#     header_img : npt.NDArray
#         The header image, thresholded, cropped, etc.

#     Returns
#     -------
#     digits : List[int]
#         The digits in the header.
#     """
#     template_imgs = [
#         cv2.imread(str(DIGIT_TEMPLATE_PATH / f"{digit}.png"), 0) for digit in range(10)
#     ]

#     digits = []
#     for digit_start_x in range(
#         0, number_img.shape[1], HEADER_PIXEL_BBOXES["digit_width"]
#     ):
#         digit_img = number_img[
#             :, digit_start_x : digit_start_x + HEADER_PIXEL_BBOXES["digit_width"]
#         ]
#         reses = [
#             np.amax(cv2.matchTemplate(digit_img, template_img, cv2.TM_CCOEFF_NORMED))
#             for template_img in template_imgs
#         ]
#         digits.append(max(range(len(reses)), key=reses.__getitem__))

#     return digits


def get_header_score(header_img: npt.NDArray) -> Union[int, None]:
    """
    Gets the score from an image of the header.

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
    ret, header_img = cv2.threshold(
        header_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    score_roi = header_img[
        HEADER_PIXEL_BBOXES["digit_start_top_left"][1] : HEADER_PIXEL_BBOXES[
            "digit_end_bottom_right"
        ][1],
        HEADER_PIXEL_BBOXES["digit_start_top_left"][0] : HEADER_PIXEL_BBOXES[
            "digit_end_bottom_right"
        ][0],
    ]

    cv2.imshow("score_roi", score_roi)  # TODO remove once this works
    # cv2.imwrite(str("number-imgs/" + str(time.time()) + ".png"), score_roi)

    # using a range as DIGIT_TEMPLATE_PATH.iterdir() returns a random order
    template_imgs = [
        cv2.imread(str(DIGIT_TEMPLATE_PATH / f"{digit}.png"), 0) for digit in range(10)
    ]

    digits = []
    for digit_start_x in range(
        0, score_roi.shape[1], HEADER_PIXEL_BBOXES["digit_width"]
    ):
        digit_img = score_roi[
            :, digit_start_x : digit_start_x + HEADER_PIXEL_BBOXES["digit_width"]
        ]

        # check if digit img is empty (or completely white due to glitched screenshot)
        # if it is, assume we reached end of score
        if np.count_nonzero(digit_img) == 0 or np.count_nonzero(digit_img == 0) == 1:
            cv2.imshow("digit_img", digit_img)
            break

        # reses = [
        #     np.amax(cv2.matchTemplate(digit_img, template_img, cv2.TM_CCOEFF_NORMED))
        #     for template_img in template_imgs
        # ]
        # # digits.append(max(range(len(reses)), key=reses.__getitem__))
        # best_digit = max(range(len(reses)), key=reses.__getitem__)

        ious = []
        for template_img in template_imgs:
            # if np.array_equal(digit_img, template_img):
            #     best_digit = template_imgs.index(template_img)
            #     break
            # may have to use sum and use float for division
            ious.append(
                np.count_nonzero(np.logical_and(digit_img, template_img))
                / np.count_nonzero(np.logical_or(digit_img, template_img))
            )
        # print(sorted(ious, reverse=True))

        best_digit = max(range(len(ious)), key=ious.__getitem__)

        # if best_digit == -1:
        #     # cv2.imwrite("/home/ani/Desktop/digit_img.png", digit_img)
        #     # cv2.imwrite("/home/ani/Desktop/template_img.png", template_imgs[0])
        #     print("Could not find digit template match.")
        #     # assume we reached end of number
        #     break

        digits.append(best_digit)

    return None if len(digits) == 0 else int("".join([str(digit) for digit in digits]))


def main():
    MONITOR_NUM = 3 if len(mss().monitors) > 3 else 0
    time.sleep(0.5)

    select_game_settings(monitor_num=MONITOR_NUM, display_bbox=False)
    time.sleep(0.5)  # wait for game to load

    # MUST BE CALLED BEFORE PRESSING PLAY
    header_bbox, gamefield_bbox = get_game_bboxes(
        monitor_num=MONITOR_NUM, display_bbox=False
    )

    # keep history of last few scores in case of screenshot glitches
    score_deque = deque(maxlen=3)
    prev_time = time.time()
    avg_fps = 0
    frame_count = 0
    with mss() as sct:
        while True:
            frame_count += 1
            header = np.array(sct.grab(header_bbox))
            gamefield = np.array(sct.grab(gamefield_bbox))

            # get the current score
            score = get_header_score(header)
            if score is not None:
                score_deque.append(score)
            else:
                # TODO: decide if we should only keep last score or a deque
                score = score_deque[-1]

            cv2.putText(
                gamefield,
                f"Score: {score}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # Displaying and counting FPS
            fps = 1 / (time.time() - prev_time)
            prev_time = time.time()
            avg_fps = (avg_fps * frame_count + fps) / (frame_count + 1)
            cv2.imshow("game", gamefield)
            cv2.imshow("header", header)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                cv2.destroyAllWindows()
                break

    print("Average FPS:", avg_fps)


if __name__ == "__main__":
    main()
