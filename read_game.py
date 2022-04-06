import time
from pathlib import Path
from typing import Iterable

import numpy as np
import numpy.typing as npt
import cv2
from mss.linux import MSS as mss
import pyautogui

# Game url:
# https://www.google.com/fbx?fbx=snake_arcade

# Values that worked on laptop
# THRESH_HEADER = np.array([[57, 174, 0], [58, 175, 255]])
# THRESH_GAMEFIELD = np.array([[45, 0, 0], [55, 255, 255]])
# For desktop:
THRESH_HEADER = np.array([[40, 0, 0], [55, 255, 255]])
THRESH_GAMEFIELD = np.array([[40, 150, 0], [45, 170, 255]])

ICON_TEMPLATE_PATH = Path("./templates").resolve()
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


def get_bbox_by_hsv(img, thresh, display_bbox=False):
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
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(img, thresh[0, :], thresh[1, :])
    masked = cv2.cvtColor(cv2.bitwise_and(img, img, mask=mask), cv2.COLOR_HSV2BGR)
    masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(masked, 0, 255, cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

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
        img = cv2.cvtColor(img.copy(), cv2.COLOR_HSV2BGR)
        cv2.drawContours(img, rects, -1, (0, 255, 0), 3)
        cv2.imshow("bbox", img)
        cv2.imshow("masked", masked)
        cv2.imshow("thresh", thresh)
        cv2.waitKey(-1)

    if len(rects) != 1:
        raise ValueError("Could not find gamefield and/or header.")

    return {
        label: val
        for label, val in zip(
            ["left", "top", "width", "height"], cv2.boundingRect(np.array(rects[0]))
        )
    }


def get_game_bboxes(monitor_num=0, display_bbox=False):
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


def relative_bbox_to_abs(bbox, monitor_num=0):
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


def relative_point_to_abs(point, monitor_num):
    """
    Converts a point relative to the monitor to an absolute point.

    Parameters
    ----------
    point : Iterable[int, int]
        The point relative to the monitor.
    monitor_num : int
        The monitor number to use. Defaults to 0, which is all monitors stitched together.

    Returns
    -------
    point : Iterable[int, int]
        The absolute point.
    """
    with mss() as sct:
        monitor = sct.monitors[monitor_num]
        return (point[0] + monitor["left"], point[1] + monitor["top"])


def find_template_match(
    template_path, match_thresh=0.9, get_center=False, monitor_num=0, display_bbox=False
):
    with mss() as sct:
        # screenshot is in BGRA format, but we need BGR
        img = np.array(sct.grab(sct.monitors[monitor_num]))[:, :, :3]

    template = cv2.imread(str(template_path))

    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)

    # check to make sure a match was found
    # TODO make this work
    if not (np.amax(res) > match_thresh):
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


def select_game_settings(monitor_num=0, display_bbox=False):
    # find and click settings icon
    settings_center = find_template_match(
        ICON_TEMPLATE_PATH / "settings-icon.png",
        match_thresh=1,
        get_center=True,
        monitor_num=monitor_num,
        display_bbox=display_bbox,
    )
    if settings_center is None:
        print("Could not find settings icon. Skipping settings selection.")
        return
    pyautogui.click(x=settings_center[0], y=settings_center[1])
    # wait for settings to load
    time.sleep(1)

    # use absolute positioning relative to settings page template match
    # to select all settings (because template match wasn't working great)
    settings_page_corner = find_template_match(
        ICON_TEMPLATE_PATH / "settings-page.png",
        get_center=False,
        monitor_num=monitor_num,
        display_bbox=display_bbox,
    )
    if settings_page_corner is None:
        print("Could not find settings page. Skipping settings selection.")
        return

    for game_setting, point in SETTINGS_PIXEL_MAPPING.items():
        pyautogui.click(
            x=point[0] + settings_page_corner[0], y=point[1] + settings_page_corner[1]
        )
        time.sleep(0.1)


def main():
    MONITOR_NUM = 3 if len(mss().monitors) > 3 else 0
    time.sleep(0.5)

    select_game_settings(monitor_num=MONITOR_NUM, display_bbox=False)

    # MUST BE CALLED BEFORE PRESSING PLAY
    header_bbox, gamefield_bbox = get_game_bboxes(
        monitor_num=MONITOR_NUM, display_bbox=True
    )

    prev_time = time.time()
    avg_fps = 0
    frame_count = 0
    with mss() as sct:
        while True:
            frame_count += 1
            header = np.array(sct.grab(header_bbox))
            gamefield = np.array(sct.grab(gamefield_bbox))

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
