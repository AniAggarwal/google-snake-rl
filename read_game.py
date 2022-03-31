import time

import numpy as np
import cv2
from mss import mss
from PIL import Image

THRESH_GAMEFIELD = np.array([[45, 0, 0], [55, 255, 255]])
THRESH_HEADER = np.array([[57, 174, 0], [58, 175, 255]])


def get_bbox_by_hsv(img, thresh, display_bbox=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(img, thresh[0, :], thresh[1, :])
    masked = cv2.cvtColor(cv2.bitwise_and(img, img, mask=mask), cv2.COLOR_HSV2BGR)
    masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(masked, 0, 255, cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        raise Exception("Could not find gamefield and/or header.")

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
        # cv2.rectangle(img, (rects[0][0][0][0], rects[0][0][0][1]), (rects[0][0][2][0], rects[0][0][2][1]), (0, 255, 0), 2)
        cv2.imshow("bbox", img)
        cv2.imshow("masked", masked)
        cv2.imshow("thresh", thresh)
        cv2.waitKey(-1)

    if len(rects) != 1:
        raise Exception("Could not find gamefield and/or header.")


    return {
        label: val
        for label, val in zip(
            ["left", "top", "width", "height"], cv2.boundingRect(np.array(rects[0]))
        )
    }


def get_game_bboxes(display_bbox=False):
    full_bbox = {"top": 0, "left": 0, "width": 1920, "height": 1080}
    sct = mss()

    img = np.array(sct.grab(full_bbox))

    header_bbox = get_bbox_by_hsv(img, THRESH_HEADER, display_bbox)
    gamefield_bbox = get_bbox_by_hsv(img, THRESH_GAMEFIELD, display_bbox)
    return header_bbox, gamefield_bbox


def select_game_settings():
    # select correct game settings
    pass

def main():
    time.sleep(1)
    header_bbox, gamefield_bbox = get_game_bboxes()
    sct = mss()

    prev_time = time.time()
    avg_fps = 0
    frame_count = 0
    while True:
        frame_count += 1
        header = np.array(sct.grab(header_bbox))
        gamefield = np.array(sct.grab(gamefield_bbox))

        # Displayign and counting FPS
        fps = 1 / (time.time() - prev_time)
        prev_time = time.time()
        avg_fps = (avg_fps * frame_count + fps) / (frame_count + 1)
        cv2.imshow("game", gamefield)
        cv2.imshow("header", header)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            cv2.destroyAllWindows()
            break

    print("Average FPS:", avg_fps)
