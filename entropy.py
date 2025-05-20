import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ignore warning
#----------------------------------------
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
#----------------------------------------
from tensorflow.python.keras.layers import VersionAwareLayers
layers = VersionAwareLayers()
import torch
from PIL import Image
import numpy as np
import tensorflow as tf
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from tta_wrapper import tta_segmentation
import cv2



from model.unet import ConvNeXt_Unet as unet
model = unet()


def preprocess_input(image, type='convnext'):
    if type != 'convnext':
        image = image / 127.5 - 1
    else:
        image = layers.Normalization(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                     variance=[(0.229 * 255) ** 2,
                                               (0.224 * 255) ** 2,
                                               (0.225 * 255) ** 2])(image)

    return image


def entropy_map(a, dim):  # [1, 256, 256]
    em = - torch.sum(a * torch.log(a + 1e-10), dim=dim)
    em /= np.log(4)
    return em


def process_image(img_path, model):
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.float32)

    # Preprocess and make predictions
    pr = model.predict(np.expand_dims(preprocess_input(img), 0))  # [1, 256, 256, 4]
    pred_u = torch.from_numpy(pr).permute(0, 3, 1, 2)  # [1, 4, 256, 256]

    # Calculate entropy
    entropy = -torch.sum(pred_u * torch.log(pred_u + 1e-10), dim=1)
    entropy /= np.log(4)  # Normalize the entropy to range [0,1]

    return entropy


def save_entropy_maps(input_folder, output_folder, model):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Traverse all .jpg files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(input_folder, filename)

            # Compute entropy map for the image
            entropy = process_image(img_path, model)

            # Save entropy map as an image
            entropy_map_path = os.path.join(output_folder, f"entropy_{filename.replace('.jpg', '.png')}")
            plt.imsave(entropy_map_path, entropy[0].numpy(), cmap='jet')  # Convert tensor to numpy array
            print(f"Saved entropy map for {filename}")


model.load_weights(r'best.h5')

# Define input and output folders
input_folder = r'synthetic\207-fid55'  # input path of synthetic images
output_folder = r'synthetic\entropy'   # output path of entropy maps

# Process all images and save entropy maps
save_entropy_maps(input_folder, output_folder, model)

