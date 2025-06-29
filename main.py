import os
import glob
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import cv2
from libs.helper_func import vid2images, images2vid
from libs.face import FaceDetector, FaceLandmarksDetector
from libs.iris import IrisDetector

def main(args):
    video_name = args.source.split('/')[-1].split('.')[0]

    if os.path.exists('./tmp'):
        shutil.rmtree('./tmp')
    os.mkdir('./tmp')
    
    vid2images(args.source, out_path='./tmp')

    if not os.path.exists('./results'):
        os.mkdir('./results')
    output_dir = './results/{}'.format(video_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(output_dir+'/images'):
        os.mkdir(output_dir+'/images')

    face_landmarks_detector = FaceLandmarksDetector()    
    iris_detector = IrisDetector()
    for image_path in tqdm(sorted(glob.glob('./tmp/*.png'))):
        input_image = np.array(Image.open(image_path).convert('RGB'))

        face_landmarks_detections = face_landmarks_detector.predict(input_image)

        for face_landmarks_detection in face_landmarks_detections:
            left_eye_image, right_eye_image, left_config, right_config = iris_detector.preprocess(input_image, face_landmarks_detection)

            left_eye_contour, left_eye_iris = iris_detector.predict(left_eye_image)
            right_eye_contour, right_eye_iris = iris_detector.predict(right_eye_image, isLeft=False)
            
            ori_left_eye_contour, ori_left_iris = iris_detector.postprocess(left_eye_contour, left_eye_iris, left_config)
            ori_right_eye_contour, ori_right_iris = iris_detector.postprocess(right_eye_contour, right_eye_iris, right_config)
            plt.imshow(input_image)
            # plt.scatter(ori_left_eye_contour[:, 0], ori_left_eye_contour[:, 1], s=3)
            plt.scatter(ori_left_iris[:, 0], ori_left_iris[:, 1], s=2)
            # plt.scatter(ori_right_eye_contour[:, 0], ori_right_eye_contour[:, 1], s=3)
            plt.scatter(ori_right_iris[:, 0], ori_right_iris[:, 1], s=2)
            plt.axis('off')
            plt.savefig(output_dir+'/images/{}'.format(image_path.split('/')[-1]))
            plt.close()

    images2vid(output_dir+'/images', output_dir=output_dir)
    if os.path.exists('./tmp'):
        shutil.rmtree('./tmp')

def live_camera():
    """Live camera iris detection"""
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)  # Use default camera
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Loading AI models... This may take a moment...")
    face_landmarks_detector = FaceLandmarksDetector()    
    iris_detector = IrisDetector()
    
    print("Models loaded successfully!")
    print("Press 'q' to quit")
    
    # Performance optimization: skip frames for faster processing
    frame_skip = 2  # Process every 2nd frame
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        frame_count += 1
        
        # Create a copy of the frame for display
        display_frame = frame.copy()
        
        # Process only every nth frame for better performance
        if frame_count % frame_skip == 0:
            try:
                # Convert BGR to RGB for processing
                input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                face_landmarks_detections = face_landmarks_detector.predict(input_image)
                
                for face_landmarks_detection in face_landmarks_detections:
                    left_eye_image, right_eye_image, left_config, right_config = iris_detector.preprocess(input_image, face_landmarks_detection)

                    left_eye_contour, left_eye_iris = iris_detector.predict(left_eye_image)
                    right_eye_contour, right_eye_iris = iris_detector.predict(right_eye_image, isLeft=False)
                    
                    ori_left_eye_contour, ori_left_iris = iris_detector.postprocess(left_eye_contour, left_eye_iris, left_config)
                    ori_right_eye_contour, ori_right_iris = iris_detector.postprocess(right_eye_contour, right_eye_iris, right_config)
                    
                    # Draw iris points on the frame
                    for point in ori_left_iris:
                        cv2.circle(display_frame, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
                    
                    for point in ori_right_iris:
                        cv2.circle(display_frame, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
                    
                    # Draw eye contours
                    for point in ori_left_eye_contour:
                        cv2.circle(display_frame, (int(point[0]), int(point[1])), 1, (255, 0, 0), -1)
                        
                    for point in ori_right_eye_contour:
                        cv2.circle(display_frame, (int(point[0]), int(point[1])), 1, (255, 0, 0), -1)
            except Exception as e:
                print(f"Processing error: {e}")
                # Continue with the next frame
        
        # Display the frame
        cv2.imshow('Live Iris Detection', display_frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

def demo():
    input_image = np.array(Image.open('./examples/01.png').convert('RGB'))
    
    face_detector = FaceDetector()
    face_detections = face_detector.predict(input_image)
    face_detector.visualize(input_image, face_detections)

    face_landmarks_detector = FaceLandmarksDetector()
    face_landmarks_detections = face_landmarks_detector.predict(input_image)
    face_landmarks_detector.visualize(input_image, face_landmarks_detections)

    for face_landmarks_detection in face_landmarks_detections:
        iris_detector = IrisDetector()
        left_eye_image, right_eye_image, left_config, right_config = iris_detector.preprocess(input_image, face_landmarks_detection)

        left_eye_contour, left_eye_iris = iris_detector.predict(left_eye_image)
        right_eye_contour, right_eye_iris = iris_detector.predict(right_eye_image, isLeft=False)

        fig, [ax1, ax2] = plt.subplots(1,2)        
        ax1.imshow(right_eye_image)
        ax1.scatter(right_eye_iris[:, 0], right_eye_iris[:, 1], s=3)
        ax1.scatter(right_eye_contour[:, 0], right_eye_contour[:, 1], s=3)
        ax1.set(title='right eye')
        ax2.imshow(left_eye_image)
        ax2.scatter(left_eye_iris[:, 0], left_eye_iris[:, 1], s=3)
        ax2.scatter(left_eye_contour[:, 0], left_eye_contour[:, 1], s=3)
        ax2.set(title='left eye')
        plt.show()
        
        ori_left_eye_contour, ori_left_iris = iris_detector.postprocess(left_eye_contour, left_eye_iris, left_config)
        ori_right_eye_contour, ori_right_iris = iris_detector.postprocess(right_eye_contour, right_eye_iris, right_config)
        plt.imshow(input_image)
        plt.scatter(ori_left_eye_contour[:, 0], ori_left_eye_contour[:, 1], s=3)
        plt.scatter(ori_left_iris[:, 0], ori_left_iris[:, 1], s=2)
        plt.scatter(ori_right_eye_contour[:, 0], ori_right_eye_contour[:, 1], s=3)
        plt.scatter(ori_right_iris[:, 0], ori_right_iris[:, 1], s=2)
        plt.show()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-s', default="", help="Path to the video")
    parser.add_argument('--camera', '-c', action='store_true', help="Use live camera for iris detection")
    args = parser.parse_args()

    if args.camera:
        live_camera()
    elif args.source == "":
        demo()
    else:
        main(args)