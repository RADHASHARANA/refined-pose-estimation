import numpy as np
import cv2

#cap = cv2.VideoCapture(input("enter file name or path: "))
cap = cv2.VideoCapture(0)
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
# termination criteria for iterative algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

cv_file = cv2.FileStorage("calib_data.yaml", cv2.FILE_STORAGE_READ)
mtx = cv_file.getNode("camera_matrix").mat()
dist = cv_file.getNode("dist_coeff").mat()
cv_file.release()


###------------------ SUBPIX ESTIMATOR ---------------------------

frame_No =0
while (True):
    ret, frame = cap.read()
    frame_No += 1

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray,5)
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpen = cv2.filter2D(blur, -1, sharpen_kernel)


        # detector parameters can be set here (List of detection parameters[3])
        parameters = cv2.aruco.DetectorParameters_create()
        parameters.adaptiveThreshConstant = 10

        # lists of ids and the corners belonging to each id
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)


        if corners:
            if len(corners)==2:
                print("\n=================== Frame no {0} =====================".format(frame_No))
                print("\nNormal Detected corners", np.array([corners[0],corners[1]]))
                subpix_corners2 = cv2.cornerSubPix(sharpen,corners[1],(5,5),(-1,-1),criteria)
                subpix_corners1 = cv2.cornerSubPix(sharpen,corners[0],(5,5),(-1,-1),criteria)
                print("\nRefined SubPix corners", np.array([subpix_corners1, subpix_corners2]))
            else :
                print("\n=================== Frame no {0} =====================".format(frame_No))
                print("\nNormal Detected corners", corners[0])
                subpix_corners1 = cv2.cornerSubPix(sharpen,corners[0],(5,5),(-1,-1),criteria)
                print("\nRefined SubPix corners", subpix_corners1)

        # font for displaying text (below)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # check if the ids list is not empty
        # if no check is added the code will crash
        if np.all(ids != None):

            #rvec, tvec ,_ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)
            #(rvec-tvec).any() # get rid of that nasty numpy value array error

            #for i in range(0, ids.size):
                # draw axis for the aruco markers
                #cv2.aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], 0.1)

            # draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)


            # code to show ids of the marker found
            strg = ''
            for i in range(0, ids.size):
                strg += str(ids[i][0])+', '

            cv2.putText(frame, "Id: " + strg, (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)


        else:
            # code to show 'No Ids' when no markers are found
            cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)

        # display the resulting frame
        cv2.imshow('frame',frame)
        #cv2.imshow("sharpen",sharpen)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


