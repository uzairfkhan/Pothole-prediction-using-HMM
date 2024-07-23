import numpy as np
import cv2
import matplotlib.pyplot as plt
from hmmlearn import hmm


# Load and preprocess images
def load_and_preprocess_image(file_path):
    img = cv2.imread(file_path)  # Load image in RGB
    img = cv2.resize(img, (256, 256))  # Resize image to 256x256 if not already
    return img


def find_focused_object(mask):
    # Find the centroid of the non-zero region in the mask
    moments = cv2.moments(mask)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        return (cx, cy)
    else:
        return (0, 0)


# Example images (replace with your own image paths)
image_paths = ["BM0001.jpg", "BM0002.jpg", "BM0003.jpg", "BM0004.jpg", "BM0005.jpg"]
masks = [load_and_preprocess_image(path) for path in image_paths]

# Extract positions of the focused object (pothole)
positions = np.array([find_focused_object(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)) for mask in masks])


# HMM to predict the next positions
def GMMHMM_Pred(input_array, n_components=3, n_iter=2000, random_state=5, n_predictions=5):
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", n_iter=n_iter, random_state=random_state)
    model.fit(input_array)
    predictions, _ = model.sample(len(input_array) + n_predictions)  # Predict next positions
    return predictions[-n_predictions:]


# Reshape positions to (n_samples, n_features) where n_features=2
reshaped_positions = positions.reshape(-1, 2)
predicted_positions = GMMHMM_Pred(reshaped_positions)

print("Predicted positions for the next 5 frames:", predicted_positions)


# Function to generate masks based on predicted positions
def generate_mask(position, image_size=(256, 256)):
    mask = np.zeros(image_size, dtype=np.uint8)
    cv2.circle(mask, (int(position[0]), int(position[1])), radius=20, color=(255, 255, 255), thickness=-1)
    return mask


# Generate the predicted RGB masks
predicted_masks = [generate_mask(pos) for pos in predicted_positions]

# Display the original and predicted masks
fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # 2 rows to accommodate 5 original and 5 predicted images
for i in range(5):
    axes[0, i].imshow(cv2.cvtColor(masks[i], cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(f'Original Mask {i + 1}')
    axes[0, i].axis('off')

for i in range(5):
    predicted_rgb_mask = cv2.merge([predicted_masks[i]] * 3)
    axes[1, i].imshow(predicted_rgb_mask)
    axes[1, i].set_title(f'Predicted Mask {i + 1}')
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()
