import cv2

# File storage in OpenCV
cv_file = cv2.FileStorage("calib_data.yaml", cv2.FILE_STORAGE_READ)

camera_matrix = cv_file.getNode("camera_matrix").mat()
dist_matrix = cv_file.getNode("dist_coeff").mat()

print("camera_matrix : \n", camera_matrix)
print("dist_matrix : \n", dist_matrix)

cv_file.release()
