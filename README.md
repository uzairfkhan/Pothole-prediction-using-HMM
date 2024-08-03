*Pothole Detection and Prediction Using HMM*

This repository contains a Python script for detecting and predicting the positions of potholes in a video. The script processes an input video, identifies the largest pothole in each frame, and uses a Hidden Markov Model (HMM) to predict future positions of the pothole. The results are saved as a collage video showing both the original frames and the predicted pothole positions.

Requirements

- Python 3.x
- NumPy
- OpenCV
- Scikit-learn
- hmmlearn

## Installation

To install the required packages, use the following command:


pip install numpy opencv-python scikit-learn hmmlearn


## Functions

### load_and_preprocess_video(video_path)

Loads and preprocesses the video. It reads the video frame by frame, resizes each frame to 256x256, and returns a list of frames.

- **Parameters**: 
  - `video_path` (str): Path to the input video.
- **Returns**: 
  - `frames` (list): List of preprocessed video frames.

### find_largest_pothole(mask, min_area=500)

Finds the largest pothole in a given mask. It identifies contours in the mask and returns the centroid of the largest contour if its area is greater than the specified minimum area.

- **Parameters**: 
  - `mask` (numpy.ndarray): Binary mask of the frame.
  - `min_area` (int, optional): Minimum area of a contour to be considered a pothole. Default is 500.
- **Returns**: 
  - `(cx, cy)` (tuple): Coordinates of the centroid of the largest pothole.

### generate_mask(position, image_size=(256, 256))

Generates a binary mask with a circle at the given position.

- **Parameters**: 
  - `position` (tuple): (x, y) coordinates of the circle's center.
  - `image_size` (tuple, optional): Size of the mask. Default is (256, 256).
- **Returns**: 
  - `mask` (numpy.ndarray): Binary mask with a circle at the specified position.

### process_video(input_video_path, output_video_path)

Main function that processes the input video, detects and predicts pothole positions, and saves the results as a collage video.

- **Parameters**: 
  - `input_video_path` (str): Path to the input video.
  - `output_video_path` (str): Path to save the output collage video.

## How It Works

1. **Loading and Preprocessing**:
   - The input video is loaded and each frame is resized to 256x256.

2. **Pothole Detection**:
   - Each frame is converted to a grayscale mask.
   - The largest pothole in each frame is identified, and its position is recorded.

3. **Position Scaling**:
   - The detected positions are scaled to a range of [0, 1] using MinMaxScaler.

4. **HMM Prediction**:
   - A Hidden Markov Model (HMM) is trained on the scaled positions to predict future positions.

5. **Mask Generation**:
   - Binary masks are generated for the predicted positions.

6. **Collage Video Creation**:
   - The original frames and the predicted masks are combined into a collage video.

## Example

To use the script, simply provide the paths for the input and output videos:

```python
input_video_path = 'input_video.mp4'
output_video_path = 'collage_pothole_predictions.avi'
process_video(input_video_path, output_video_path)

print("Collage video saved as:", output_video_path)
```

## Notes

- Ensure the input video path is correct.
- The output video will be saved in the same directory as the script, unless a different path is specified.
