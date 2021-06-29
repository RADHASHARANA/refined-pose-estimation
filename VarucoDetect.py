#from imutils.video import VideoStream
#import argparse
#import imutils
import time
import cv2

def getCameraMatrix():
    cv_file = cv2.FileStorage("calib_data.yaml", cv2.FILE_STORAGE_READ)
    camera_matrix = cv_file.getNode("camera_matrix").mat()
    cv_file.release()
    return camera_matrix

def getDistortionCoeff():
    cv_file = cv2.FileStorage("calib_data.yaml", cv2.FILE_STORAGE_READ)
    dist_coeff = cv_file.getNode("dist_coeff").mat()
    cv_file.release()
    return dist_coeff


# load the ArUCo dictionary and grab the ArUCo parameters
def loadDict():
    print("[INFO] detecting ArUco tags.....")
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    arucoParams = cv2.aruco.DetectorParameters_create()
    return [arucoDict, arucoParams]

# initializing the video stream and allowing the camera sensor to warm up
def liveVideoStream():
    print("[INFO] starting video stream....")
    vs = cv2.VideoCapture(0)
    #vs = VideoStream(src=0).start()
    time.sleep(2.0)
    return vs

def staticVideoStream():
    path=input("enter file name or path: ")
    #vs = VideoStream(src = path).start()
    cap = cv2.VideoCapture(path)
    #time.sleep(2.0)
    return cap


"""def getContour(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #define area for square markers
    AREA = frame.shape[0]*frame.shape[1]/4
    if len(contours) > 0:
        for contour in contours:

            approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)

            if len(approx)==4:
                x, y, w, h = cv2.boundingRect(approx)

                if cv2.contourArea(contour) < AREA and cv2.contourArea(contour) > 1800 :
                    print(AREA, frame.shape[0], frame.shape[1])
                    cv2.putText(frame, "detected", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                    cv2.drawContours(frame, [contour], -1, (0,255,0), 2, cv2.LINE_AA)
                    markerContour = contour.reshape((len(contour), 2))
                    print(markerContour.tolist())

        #Display the resulting frame
        cv2.imshow('frame',frame)"""

def detectTags(vs, arucoDict, arucoParams):
# loop over the frames from the video stream
    frame_no = 0
    while True:
        ret, frame = vs.read()
        if ret :
            frame_no +=1
            #frame = imutils.resize(frame, width=900)

            # detect ArUco markers in the input frame
            (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
            #getContour(frame)
            # verify *at least* one ArUco marker was detected
            if len(corners) > 0:
                # flatten the ArUco IDs list
                ids = ids.flatten()
                #print(corners)
                print("\n======================= FRAME {0} =========================".format(frame_no))
                # loop over the detected ArUCo corners
                for (markerCorner, markerID) in zip(corners, ids):
                    # extract the marker corners (which are always returned
                    # in top-left, top-right, bottom-right, and bottom-left order)
                    corners = markerCorner.reshape((4, 2))
                    print("corners of marker ID {0} are :\n {1} \n\n------------------".format(markerID,corners))
                    (topLeft, topRight, bottomRight, bottomLeft) = corners
                    # convert each of the (x, y)-coordinate pairs to integers
                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))
                    # draw the bounding box of the ArUCo detection
                    cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
                    cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
                    cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
                    cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)
                    # compute and draw the center (x, y)-coordinates of the
                    # ArUco marker
                    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                    cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                    cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
                    # draw the ArUco marker ID on the frame
                    cv2.putText(frame, str(markerID), (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
        else:
            break


######################### DRIVER CODE ##########################
dictParams = loadDict()

camera_matrix = getCameraMatrix()
dist_coeff = getDistortionCoeff()

camera_matrix_L = camera_matrix.tolist()
dist_coeff_L = dist_coeff.tolist()

print(camera_matrix, dist_coeff)
#print(camera_matrix_L, dist_coeff_L)

source_type = input("enter your choice : \n1. Live Stream \n2. Static Video \nYour Choice: ")
vs = liveVideoStream() if source_type == "1"  else staticVideoStream()

detectTags(vs, dictParams[0], dictParams[1])
cv2.destroyAllWindows()
vs.release()
