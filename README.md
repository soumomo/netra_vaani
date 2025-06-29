# Real-Time Iris Gaze and Blink Detection

This project uses MediaPipe and OpenCV to perform real-time iris tracking for gaze direction estimation and robust blink detection from a webcam feed.

## Features

- **Gaze Direction Estimation:** Detects if the user is looking LEFT, RIGHT, or CENTER.
- **Blink Detection:** Accurately counts user blinks using the Eye Aspect Ratio (EAR) method.
- **Real-time Performance:** Processes webcam feed with low latency.
- **Robustness:** Gaze detection is paused during blinks to prevent noisy data and false readings.

## Setup and Usage

### Prerequisites

- Python 3.7+
- A connected webcam

### Installation

1.  **Clone the repository (or download the source code).**

2.  **Navigate to the project directory:**
    ```bash
    cd MediaPipe_Iris-main
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Activate the virtual environment:**

    *On Linux/macOS:*
    ```bash
    source iris_venv/bin/activate
    ```

    *On Windows:*
    ```bash
    .\iris_venv\Scripts\activate
    ```

2.  **Execute the main script:**
    ```bash
    python original_iris_tracker.py
    ```

3.  A window will open showing your webcam feed with the gaze direction and blink count displayed.

4.  **To quit the application, press the 'q' key.**
