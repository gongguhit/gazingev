import cv2
import numpy as np

# Initialize ArUco detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Open video file
video = cv2.VideoCapture('aruco_test.mp4')

# Camera calibration parameters
camera_matrix = np.array([[8.94185389e+02, 0.00000000e+00, 1.28203484e+03],
 [0.00000000e+00, 8.92035753e+02, 7.15336290e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)
dist_coeffs = np.zeros((4,1))

# Define the marker length (in meters)
marker_length = 0.05

# Define the 3D points of the marker in its own coordinate system
objPoints = np.array([
    [-marker_length/2, marker_length/2, 0],
    [marker_length/2, marker_length/2, 0],
    [marker_length/2, -marker_length/2, 0],
    [-marker_length/2, -marker_length/2, 0]
], dtype=np.float32)

def draw_thick_axes(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype(int))
    img = cv2.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (0,0,255), 3)  # X-axis (Red)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 3)  # Y-axis (Green)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (255,0,0), 3)  # Z-axis (Blue)
    return img

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    # Detect ArUco markers
    corners, ids, rejected = detector.detectMarkers(frame)
    
    if ids is not None:
        # Draw bounding boxes
        cv2.aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))
        
        for i in range(len(ids)):
            # Get the corners of the current marker
            markerCorners = corners[i][0]
            
            # Estimate pose for the current marker
            success, rvec, tvec = cv2.solvePnP(objPoints, markerCorners, camera_matrix, dist_coeffs)
            
            if success:
                # Define axis points (make them longer)
                axis_length = marker_length * 2  # Increase this factor to make axes longer
                axis_points = np.float32([[axis_length,0,0], [0,axis_length,0], [0,0,axis_length]])
                
                # Project 3D points to image plane
                imgpts, jac = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
                
                # Draw thick axes
                frame = draw_thick_axes(frame, markerCorners, imgpts)
    
    # Display the frame
    cv2.imshow('ArUco Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()