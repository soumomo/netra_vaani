import cv2
import numpy as np
import mediapipe as mp
from math import hypot, atan2, degrees
import time

class IrisGazeDetector:
    def __init__(self, smoothing_window=7, movement_threshold=0.025):
        self.smoothing_window = smoothing_window
        self.iris_history = []
        self.movement_threshold = movement_threshold  # Adjusted for better sensitivity
        self.current_direction = "CENTER"
        self.debug_info = ""
        
        # MediaPipe setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmarks indices for MediaPipe Face Mesh
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
    
    def detect_gaze_direction(self, image):
        # Convert the BGR image to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]
        
        # Process the image and get the face landmarks
        results = self.face_mesh.process(img_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get eye contours
                left_eye_contour = self._get_eye_contour(face_landmarks, self.LEFT_EYE, img_w, img_h)
                right_eye_contour = self._get_eye_contour(face_landmarks, self.RIGHT_EYE, img_w, img_h)
                
                # Get iris landmarks
                left_iris_landmarks = self._get_iris_landmarks(face_landmarks, self.LEFT_IRIS, img_w, img_h)
                right_iris_landmarks = self._get_iris_landmarks(face_landmarks, self.RIGHT_IRIS, img_w, img_h)
                
                # Calculate relative positions
                left_eye_pos = self.get_eye_relative_position(left_iris_landmarks, left_eye_contour, img_w, img_h)
                right_eye_pos = self.get_eye_relative_position(right_iris_landmarks, right_eye_contour, img_w, img_h)
                
                if left_eye_pos and right_eye_pos:
                    # Average the positions of both eyes
                    avg_relative_x = (left_eye_pos[0] + right_eye_pos[0]) / 2
                    
                    # Add to history for smoothing
                    self.iris_history.append(avg_relative_x)
                    if len(self.iris_history) > self.smoothing_window:
                        self.iris_history.pop(0)
                    
                    # Calculate smoothed position
                    smooth_x = sum(self.iris_history) / len(self.iris_history)
                    
                    # Determine direction based on position relative to center
                    if smooth_x < 0.5 - self.movement_threshold:
                        self.current_direction = "LEFT"
                    elif smooth_x > 0.5 + self.movement_threshold:
                        self.current_direction = "RIGHT"
                    else:
                        self.current_direction = "CENTER"
                    
                    # Debug info
                    self.debug_info = f"Gaze: {self.current_direction} | Pos: {smooth_x:.3f}"
                    
                    # Draw debug visualization
                    self._draw_visuals(image, left_iris_landmarks, right_iris_landmarks, 
                                     left_eye_contour, right_eye_contour, img_w, img_h)
        
        return self.current_direction, image
    
    def _get_eye_contour(self, face_landmarks, eye_indices, img_w, img_h):
        """Get the contour points for an eye."""
        return [
            (int(face_landmarks.landmark[idx].x * img_w), 
             int(face_landmarks.landmark[idx].y * img_h))
            for idx in eye_indices
        ]
    
    def _get_iris_landmarks(self, face_landmarks, iris_indices, img_w, img_h):
        """Get the iris landmarks."""
        return [
            (int(face_landmarks.landmark[idx].x * img_w), 
             int(face_landmarks.landmark[idx].y * img_h))
            for idx in iris_indices
        ]
    
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
    
    def _draw_visuals(self, image, left_eye_landmarks, right_eye_landmarks, left_contour, right_contour, img_w, img_h):
        """Draw visualization of iris detection and direction."""
        # Draw contours and landmarks
        if left_contour: cv2.polylines(image, [np.array(left_contour, np.int32)], True, (0, 255, 0), 1)
        if right_contour: cv2.polylines(image, [np.array(right_contour, np.int32)], True, (0, 255, 0), 1)
        if left_eye_landmarks:
            for (x, y) in left_eye_landmarks: 
                if 0 <= x < img_w and 0 <= y < img_h: cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
        if right_eye_landmarks:
            for (x, y) in right_eye_landmarks:
                if 0 <= x < img_w and 0 <= y < img_h: cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
        
        # Draw direction text
        cv2.putText(image, self.debug_info, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Direction: {self.current_direction}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw direction arrow
        arrow = ""
        if self.current_direction == "LEFT":
            arrow = "<--"
        elif self.current_direction == "RIGHT":
            arrow = "-->"
        elif self.current_direction == "CENTER":
            arrow = "-O-"

        (text_width, text_height), _ = cv2.getTextSize(arrow, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
        cv2.putText(image, arrow, (int(img_w/2 - text_width/2), 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

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
    print("Starting Iris Gaze Detector")
    print("1. Keep your head straight")
    print("2. Look at the camera")
    print("3. The system will calibrate automatically")
    print("4. Move your eyes to control the direction")
    print("5. Press 'q' to quit")
    
    # Try to open camera
    cap = None
    for i in range(3):  # Try first 3 camera indices
        cap = try_open_camera(i)
        if cap is not None:
            print(f"Using camera {i}")
            break
    
    if cap is None:
        print("Error: Could not open any camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Create detector
    detector = IrisGazeDetector()
    
    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Flip frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect gaze direction
        direction, frame = detector.detect_gaze_direction(frame)
        
        # Show the frame
        cv2.imshow('Iris Gaze Detector', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
