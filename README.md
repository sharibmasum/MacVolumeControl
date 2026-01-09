## Hand Gesture Volume Control (macOS)

This project is a real-time computer vision application that allows you to control your system volume using hand gestures captured from a webcam. It uses MediaPipe for hand tracking and OpenCV for video processing.

Right hand: controls volume using the distance between thumb and index finger  
Left hand: toggles a HOLD state (freezes/unfreezes volume control)  
Runs in real time with visual feedback (FPS, volume level, hold state)

## Requirements

- macOS (uses osascript for system volume control)
- Python 3.9+

## Clone the repository

git clone https://github.com/sharibmasum/MacVolumeControl.git  

## Install dependencies

pip install opencv-python mediapipe numpy

## Run the project

python main.py

## Controls

Right hand  
- Move thumb and index finger closer to decrease volume  
- Move thumb and index finger farther apart to increase volume  

Left hand  
- Bring left hand into view to toggle HOLD  
- Remove and reintroduce left hand to toggle again  

Press `q` to exit the application.
