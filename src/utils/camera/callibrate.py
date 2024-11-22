import numpy as np
import cv2
import json
import time
import argparse
from pathlib import Path
import EasyPySpin


def calibrate_camera(video_path=None):
    # Checkerboard parameters
    CHECKERBOARD = (12, 8)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Object points
    objpoints = []
    imgpoints = []
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    # Video capture setup
    cap = cv2.VideoCapture(0 if video_path is None else video_path)
    
    start_time = time.time()
    duration = 20  # 20 seconds for recording
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 5 != 0:  # Process every 5th frame
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            objpoints.append(objp)
            
            # Draw corners
            cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret)

        cv2.imshow('Calibration', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q') or \
           (video_path is None and time.time() - start_time > duration):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Calculate camera calibration
    if imgpoints:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )

        # Save calibration results
        calib_result = {
            'camera_matrix': mtx.tolist(),
            'dist_coeffs': dist.tolist()
        }
        
        output_path = Path(__file__).parent / 'camera_calibration.json'
        with open(output_path, 'w') as f:
            json.dump(calib_result, f, indent=4)
        
        return True
    return False

def get_camera_params(calib_file=None):
    """
    Extract camera parameters (fx, fy, cx, cy) from calibration file.
    Returns: tuple (fx, fy, cx, cy)
    """
    if calib_file is None:
        calib_file = Path(__file__).parent / 'camera_calibration.json'
    
    with open(calib_file, 'r') as f:
        calib_data = json.load(f)
    
    camera_matrix = calib_data['camera_matrix']
    fx = camera_matrix[0][0]  # Focal length x
    fy = camera_matrix[1][1]  # Focal length y
    cx = camera_matrix[0][2]  # Principal point x
    cy = camera_matrix[1][2]  # Principal point y
    
    return fx, fy, cx, cy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, help='Path to video file (optional)')
    args = parser.parse_args()
    
    success = calibrate_camera(args.video)
    if success:
        print("Calibration completed successfully!")
        fx, fy, cx, cy = get_camera_params()
        print(f"Camera parameters:")
        print(f"fx: {fx:.2f}, fy: {fy:.2f}")
        print(f"cx: {cx:.2f}, cy: {cy:.2f}")
    else:
        print("Calibration failed! No checkerboard patterns detected.")
