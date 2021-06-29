import numpy as np
import argparse
import cv2

# constructing the argument parse and parsing he aruments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--id", type=int, required=True, help="ID of ArUco tag to generate")
args = vars(ap.parse_args())


# loading the Dictionary
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)

# finally Generating the ArUco markers with OpenCV
print("[INFO] generating ArUco tag '{}'".format(args["id"]))
print("\n============ PRESS ENTER to EXIT ============\n")
tag = np.zeros((400, 400, 1), dtype="uint8")
cv2.aruco.drawMarker(arucoDict, args["id"], 400, tag, 1)

cv2.imshow("ArUco Tag", tag)
cv2.waitKey(0)
