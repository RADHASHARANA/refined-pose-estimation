import numpy as np
import cv2

cap = cv2.VideoCapture(0)
frame_No = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame_No += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray,5)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

    thresh = cv2.threshold(sharpen, 127, 255, cv2.THRESH_BINARY_INV)[1]

    contours = cv2.findContours(thresh , cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    #Detect aruco tags
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)


    polyDPcorners = []

    if corners:

        #accuracy to which aruco match is required
        coordinate_accuracy = 2

        corners_array = np.array(corners)
        corners_int = corners_array.astype(int)
        corner_coordinates_array = np.vstack(np.vstack(corners_int))
        corner_coordinates_split_array = np.hsplit(corner_coordinates_array, 2)
        corner_x = corner_coordinates_split_array[0].tolist()
        corner_y = corner_coordinates_split_array[1].tolist()

        for contour in contours:

            flat_contour = contour.flatten()
            flat_contour_set = set(flat_contour)


            if 0 not in flat_contour_set and frame.shape[0] not in flat_contour_set  and frame.shape[0]-1 not in flat_contour_set and frame.shape[1] not in flat_contour_set and frame.shape[1]-1 not in flat_contour_set:

                approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)

                if len(approx)==4:

                    x, y, w, h = cv2.boundingRect(approx)

                    #define area for square markers
                    AREA = frame.shape[0]*frame.shape[1]/4

                    if cv2.contourArea(contour) < AREA and cv2.contourArea(contour) > 900 :

                        contour_coordinates_array = np.vstack(contour)
                        contour_coordinates_split_array = np.hsplit(contour_coordinates_array,2)
                        contour_x = contour_coordinates_split_array[0].tolist()
                        contour_y = contour_coordinates_split_array[1].tolist()


                        contour_list = contour.tolist()

                        match_x = []
                        match_y = []
                        for x_coord in corner_x :
                            if x_coord in contour_x:
                                match_x.append(x_coord)
                        for y_coord in corner_y :
                            if y_coord in contour_y:
                                match_y.append(y_coord)


                        if len(match_x) >= coordinate_accuracy and len(match_y) >= coordinate_accuracy:
                            cv2.putText(frame, "detected", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                            cv2.drawContours(frame, [contour], -1, (0,255,0), 2, cv2.LINE_AA)
                            approx_1 = np.vstack(approx)
                            approx_L = [approx_1]
                            approx_2 = np.array(approx_L)
                            polyDPcorners.append(approx_2.astype(float))

                            #print("contour List\n", contour_list)
                            #print("corners\n", corners)
    if polyDPcorners:
        print("\n================ Frame No {0} ================".format(frame_No))
        print("\nshape fit Refined corners \n",polyDPcorners)
        print("\nNormal Detected corners \n",corners)

    #Display the resulting frame
    cv2.imshow('frame',frame)
    #cv2.imshow('thresh',thresh)
    #cv2.imshow('sharpen',sharpen)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
