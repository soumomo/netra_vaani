import cv2
import mediapipe as mp
import numpy as np
from math import hypot, atan2, degrees
import time
import random

class HeadPoseEstimator:
    def __init__(self):
        # Define the 3D model points for the face
        self.face_3d = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)
        
        # Define the 2D image points for the face
        self.face_2d = np.zeros((6, 2), dtype=np.float64)
    
    def update(self, face_landmarks, img_w, img_h):
        # Update 2D points from face landmarks
        self.face_2d[0] = [face_landmarks.landmark[1].x * img_w, face_landmarks.landmark[1].y * img_h]  # Nose tip
        self.face_2d[1] = [face_landmarks.landmark[152].x * img_w, face_landmarks.landmark[152].y * img_h]  # Chin
        self.face_2d[2] = [face_landmarks.landmark[33].x * img_w, face_landmarks.landmark[33].y * img_h]  # Left eye left corner
        self.face_2d[3] = [face_landmarks.landmark[263].x * img_w, face_landmarks.landmark[263].y * img_h]  # Right eye right corner
        self.face_2d[4] = [face_landmarks.landmark[61].x * img_w, face_landmarks.landmark[61].y * img_h]  # Left mouth corner
        self.face_2d[5] = [face_landmarks.landmark[291].x * img_w, face_landmarks.landmark[291].y * img_h]  # Right mouth corner
        
        # Camera matrix
        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype=np.float64
        )
        
        # Distortion coefficients
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        
        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            self.face_3d, self.face_2d, camera_matrix, dist_coeffs)
        
        if not success:
            return 0, 0, 0
            
        # Get rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)
        
        # Get angles
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
        # Get the y rotation degree
        x_angle = angles[0] * 360
        y_angle = angles[1] * 360
        z_angle = angles[2] * 360
        
        return x_angle, y_angle, z_angle

class EyeTracker:
    def __init__(self, smoothing_window=5):
        """
        Simple eye tracker with red ball cursor.
        """
        self.smoothing_window = smoothing_window
        self.iris_history = []
        
        # MediaPipe setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Cursor properties
        self.cursor_pos = None
        self.cursor_radius = 20
        self.cursor_color = (0, 0, 255)  # Red
        
        # Screen bounds
        self.screen_width = 0
        self.screen_height = 0
        
        # Calibration
        self.calibrating = True
        self.calibration_points = []
        self.calibration_targets = [
            (0.2, 0.5),  # Left
            (0.8, 0.5),  # Right
            (0.5, 0.5),  # Center
            (0.5, 0.2),  # Top
            (0.5, 0.8)   # Bottom
        ]
        self.current_target = 0
        self.calibration_data = []
        self.calibration_complete = False
        
    def get_eye_relative_position(self, eye_landmarks, eye_contour, img_w, img_h):
        """
        Calculate the relative position of the iris within the eye contour.
        Returns (relative_x, relative_y) where (0,0) is top-left and (1,1) is bottom-right of the eye.
        """
        if not eye_landmarks or not eye_contour:
            return None
            
        # Convert to numpy array if not already
        eye_landmarks_np = np.array(eye_landmarks)
        eye_contour_np = np.array(eye_contour)
        
        # Get the bounding box of the eye contour
        x_min, y_min = np.min(eye_contour_np, axis=0)
        x_max, y_max = np.max(eye_contour_np, axis=0)
        eye_width = x_max - x_min
        eye_height = y_max - y_min
        
        if eye_width == 0 or eye_height == 0:
            return (0.5, 0.5)  # Return center if eye dimensions are invalid
        
        # Calculate center of the iris
        iris_center = np.mean(eye_landmarks_np, axis=0)
        
        # Calculate relative position within the eye
        relative_x = (iris_center[0] - x_min) / eye_width
        relative_y = (iris_center[1] - y_min) / eye_height
        
        return (relative_x, relative_y)
    
    def get_eye_contour(self, face_landmarks, eye_indices, img_w, img_h):
        """Get the contour points for an eye."""
        return [
            (int(face_landmarks.landmark[idx].x * img_w), 
             int(face_landmarks.landmark[idx].y * img_h))
            for idx in eye_indices
        ]
    
    def advance_calibration(self, frame, x, y):
        """Handle the calibration process."""
        if self.current_target >= len(self.calibration_targets):
            self.calibrating = False
            self.calibration_complete = True
            return
            
        target_x, target_y = self.calibration_targets[self.current_target]
        
        # Draw target
        target_screen_x = int(target_x * self.screen_width)
        target_screen_y = int((1 - target_y) * self.screen_height)
        cv2.circle(frame, (target_screen_x, target_screen_y), 10, (0, 255, 0), -1)
        
        # Check if we're close enough to the target
        dx = x - target_x
        dy = y - target_y
        distance = (dx*dx + dy*dy) ** 0.5
        
        if distance < 0.1:  # Threshold for considering the target reached
            self.calibration_data.append((x, y, target_x, target_y))
            self.current_target += 1
            
            # If calibration complete, calculate mapping
            if self.current_target >= len(self.calibration_targets):
                self._finalize_calibration()

    def _calculate_thresholds(self):
        """Calculates gaze thresholds based on calibrated positions."""
        center = self.calibrated_positions.get('center', 0.5)
        left = self.calibrated_positions.get('left', 0.3)
        right = self.calibrated_positions.get('right', 0.7)

        self.left_threshold = (center + left) / 2
        self.right_threshold = (center + right) / 2
        print(f"Calibration complete. Positions: {self.calibrated_positions}")
        print(f"New thresholds: Left={self.left_threshold:.3f}, Right={self.right_threshold:.3f}")

    def process_frame(self, frame):
        """Process a frame and return the frame with visualization."""
        if self.screen_width == 0:
            self.screen_height, self.screen_width = frame.shape[:2]
        
        # Convert the BGR image to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image and get the face landmarks
        results = self.face_mesh.process(img_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Define eye contours and iris indices
                left_iris_indices = [474, 475, 476, 477]  # Left eye iris landmarks
                right_iris_indices = [469, 470, 471, 472]  # Right eye iris landmarks
                
                # Get iris positions
                left_iris = self._get_iris_position(face_landmarks, left_iris_indices)
                right_iris = self._get_iris_position(face_landmarks, right_iris_indices)
                
                if left_iris and right_iris:
                    # Average the iris positions
                    avg_x = (left_iris[0] + right_iris[0]) / 2
                    avg_y = (left_iris[1] + right_iris[1]) / 2
                    
                    # Add to history for smoothing
                    self.iris_history.append((avg_x, avg_y))
                    if len(self.iris_history) > self.smoothing_window:
                        self.iris_history.pop(0)
                    
                    # Calculate smoothed position
                    if self.iris_history:
                        smooth_x = sum(p[0] for p in self.iris_history) / len(self.iris_history)
                        smooth_y = sum(p[1] for p in self.iris_history) / len(self.iris_history)
                        
                        # Map to screen coordinates (invert y-axis)
                        screen_x = int(smooth_x * self.screen_width)
                        screen_y = int((1 - smooth_y) * self.screen_height)
                        
                        # Update cursor position
                        self.cursor_pos = (screen_x, screen_y)
                        
                        # Handle calibration if needed
                        if self.calibrating:
                            self.advance_calibration(frame, smooth_x, smooth_y)
        
        # Draw the cursor if position is available
        if self.cursor_pos:
            cv2.circle(frame, self.cursor_pos, self.cursor_radius, self.cursor_color, -1)
        
        # Draw calibration UI
        self._draw_ui(frame)
        
        return frame

    def _get_iris_position(self, face_landmarks, iris_indices):
        """Calculate the normalized position of the iris."""
        points = []
        for idx in iris_indices:
            landmark = face_landmarks.landmark[idx]
            points.append((landmark.x, landmark.y))
        
        if points:
            avg_x = sum(p[0] for p in points) / len(points)
            avg_y = sum(p[1] for p in points) / len(points)
            return (avg_x, avg_y)
        return None
        
    def _finalize_calibration(self):
        """Calculate the mapping from eye position to screen coordinates."""
        # Simple linear mapping for now - could be enhanced with more sophisticated calibration
        self.calibration_complete = True
        self.calibrating = False
        print("Calibration complete!")

    def _draw_ui(self, frame):
        """Draw the user interface elements."""
        if self.calibrating:
            target_x, target_y = self.calibration_targets[self.current_target]
            target_screen_x = int(target_x * self.screen_width)
            target_screen_y = int((1 - target_y) * self.screen_height)
            
            # Draw instructions
            cv2.putText(frame, "Look at the green circle", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Calibration {self.current_target + 1}/{len(self.calibration_targets)}", 
                       (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw debug info
        if self.cursor_pos:
            cv2.putText(frame, f"X: {self.cursor_pos[0]}, Y: {self.cursor_pos[1]}", 
                       (50, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def try_open_camera(index):
    """Try to open a camera with the given index."""
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        # Try to read a frame to confirm the camera works
        ret, _ = cap.read()
        if ret:
            return cap
        cap.release()
    return None

def main():
    print("Starting Eye Tracker with Red Ball Cursor")
    print("1. Keep your head still")
    print("2. Follow the on-screen calibration")
    print("3. Move only your eyes to control the red ball")
    print("4. Press 'q' to quit")
    
    # Initialize the eye tracker
    tracker = EyeTracker()
    
    # Try to open a camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Flip frame for selfie view
            frame = cv2.flip(frame, 1)
            
            # Process the frame
            frame = tracker.process_frame(frame)
            
            # Display the frame
            cv2.imshow('Eye Tracker - Press Q to quit', frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
