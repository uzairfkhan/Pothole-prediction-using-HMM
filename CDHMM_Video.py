import numpy as np
import cv2
from sklearn.preprocessing import MinMaxScaler
from hmmlearn import hmm


# Load and preprocess a video
def load_and_preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (256, 256))  # Resize frame to 256x256 if not already
        frames.append(frame)
    cap.release()
    return frames


def find_largest_pothole(mask, min_area=500):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0)

    # Filter contours based on area and find the largest one
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) > min_area:
        M = cv2.moments(largest_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy)
    return (0, 0)


def generate_mask(position, image_size=(256, 256)):
    mask = np.zeros(image_size, dtype=np.uint8)
    cv2.circle(mask, (int(position[0]), int(position[1])), radius=20, color=(255, 255, 255), thickness=-1)
    return mask


def process_video(input_video_path, output_video_path):
    # Load video
    frames = load_and_preprocess_video(input_video_path)
    num_frames = len(frames)

    # Extract positions of the largest pothole for initial frames
    positions = []
    for frame in frames:
        mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        position = find_largest_pothole(mask)
        positions.append(position)

    positions = np.array(positions)

    # Handle cases with no positions
    if len(positions) == 0:
        print("No potholes detected.")
        return

    # Scale the positions
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_positions = scaler.fit_transform(positions)

    # HMM to predict the next positions
    def CDHMM_Pred(input_array, n_components=3, n_iter=1000, random_state=9):
        model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", n_iter=n_iter,
                                random_state=random_state)
        model.fit(input_array)
        predictions, _ = model.sample(len(input_array))  # Predict positions
        return predictions

    predicted_scaled_positions = CDHMM_Pred(scaled_positions)
    predicted_positions = scaler.inverse_transform(predicted_scaled_positions)

    # Generate the predicted RGB masks
    predicted_masks = [generate_mask(pos) for pos in predicted_positions]

    # Create a VideoWriter object to save the collage as a video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 30,
                                   (512, 256))  # Width is 512 (256 + 256) for the collage

    for i in range(num_frames):
        original_frame = frames[i]
        predicted_frame = generate_mask(positions[i])
        predicted_rgb_frame = cv2.merge([predicted_frame] * 3)

        # Create a collage by concatenating original and predicted frames
        collage = np.hstack((original_frame, predicted_rgb_frame))

        video_writer.write(collage)

    # Release the VideoWriter object
    video_writer.release()


# Process the video
input_video_path = 'input_video.mp4'
output_video_path = 'collage_pothole_predictions.avi'
process_video(input_video_path, output_video_path)

print("Collage video saved as:", output_video_path)
