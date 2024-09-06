import numpy as np
import cv2
import os

# Define the dimensions of the chessboard (internal corners)
CHECKERBOARD = (7, 9)  # 7x9 squares means 6x8 internal corners
SQUARE_SIZE = 0.02  # 20mm = 0.02m

# Prepare object points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE  # Scale to real-world dimensions

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Directory containing calibration images
calibration_dir = './'  # Adjust this path as needed

# Variable to store image size
img_size = None

# Iterate through all images in the directory
for filename in os.listdir(calibration_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Add or remove file extensions as needed
        filepath = os.path.join(calibration_dir, filename)
        img = cv2.imread(filepath)
        
        if img is None:
            print(f"Failed to read image: {filename}")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Store the image size (assuming all images are the same size)
        if img_size is None:
            img_size = gray.shape[::-1]

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners (optional, for verification)
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
        else:
            print(f"Chessboard not found in image: {filename}")

cv2.destroyAllWindows()

# Check if we have enough images for calibration
if len(objpoints) == 0:
    print("No valid chessboard images found. Calibration failed.")
else:
    print(f"Found {len(objpoints)} valid chessboard images.")
    
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # Print camera calibration parameters
    print("\nCamera matrix:")
    print(mtx)
    print("\nDistortion coefficients:")
    print(dist)

    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print(f"\nTotal error: {mean_error / len(objpoints)}")

    # Save the camera calibration results
    np.savez('camera_calibration.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    print("Calibration results saved to 'camera_calibration.npz'")