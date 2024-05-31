import os
import random
import cv2
import torch
from network import SortingNet
import torchvision.transforms as transforms

def load_random_image(folder_path):
    """
    Loads a random image from the specified folder using OpenCV.

    Parameters:
    folder_path (str): The path to the folder containing images.

    Returns:
    numpy.ndarray: The loaded image array.
    """
    # Get a list of files in the folder
    files = os.listdir(folder_path)
    
    # Filter out non-image files based on common image file extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif')
    image_files = [file for file in files if file.lower().endswith(image_extensions)]
    
    # Check if there are any image files in the folder
    if not image_files:
        raise FileNotFoundError("No image files found in the specified folder.")
    
    # Select a random image file
    random_image_file = random.choice(image_files)
    
    # Construct the full path to the image file
    image_path = os.path.join(folder_path, random_image_file)
    
    # Load the image using cv2
    image = cv2.imread(image_path)
    
    if image is None:
        raise IOError(f"Failed to load the image file: {image_path}")
    
    return image

def transform_image(image):
    height, width = image.shape[:2]

    def rotate_image(image, angle):
        M = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1)
        return cv2.warpAffine(image, M, (width, height))

    def mirror_image(image):
        return cv2.flip(image, 1)

    transformations = []
    
    for angle in [0, 90, 180, 270]:
        rotated_image = rotate_image(image, angle)
        mirrored_image = mirror_image(rotated_image)
        for img in [rotated_image, mirrored_image]:
            transformations.append(img)

    return transformations

def preprocess_image(image):
    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply the transformations to the image
    image_tensor = transform(image)

    # Add a batch dimension to the tensor
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor

# Load the SortingNet network
network = SortingNet()

# Specify the path to the folder containing images
folder_path = "/home/scferro/Documents/final_project/training_images"

image_tensors = []
label_tensors = []

# Main training loop
while True:
    # Load a random image from the folder
    random_image = load_random_image(folder_path)

    # Display the transformed image
    cv2.imshow('Transformed Image', random_image)
    cv2.waitKey(0)  # Wait for a key press to close the image window

    # Create tensor
    random_image_tensor = preprocess_image(random_image)
    
    # Apply transformations to the image
    transformed_images = transform_image(random_image)
    
    # Get the prediction from the network
    prediction = network.forward(random_image_tensor)
    print(f"Prediction: {prediction.item():.4f}")
    
    # Prompt for user input for the correct label
    while True:
        try:
            label = int(input("Enter the correct label (0 or 1): "))
            if label in [0, 1]:
                break
            else:
                print("Invalid input. Please enter 0 or 1.")
        except ValueError:
            print("Invalid input. Please enter 0 or 1.")
    
    # Close the image window
    cv2.destroyAllWindows()

    # Convert to tensors
    for img_tf in transformed_images:
        img_tensor = preprocess_image(img_tf)
        label_tensor = torch.tensor([[label]], dtype=torch.float32)
        image_tensors.append(img_tensor)
        label_tensors.append(label_tensor)

    # Limit length of lists to 100 items
    if len(image_tensors) > 100:
        image_tensors = image_tensors[-100:]
        label_tensors = label_tensors[-100:]
    
    # Train the network with the image and user-provided label
    network.train_network(image_tensors, label_tensors)