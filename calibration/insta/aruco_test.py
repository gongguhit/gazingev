import cv2
import numpy as np

# Initialize ArUco detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Open video file
video = cv2.VideoCapture('aruco.mp4')

# Camera calibration parameters (replace with your own values)
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
                # Draw axis for the marker
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_length/2)
    
    # Display the frame
    cv2.imshow('ArUco Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()