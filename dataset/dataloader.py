import math
import os
from random import shuffle
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms
from utils import gray2rgb, preprocess_input
import random
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
from multiprocessing import Pool



class UnetDataset(tf.keras.utils.Sequence):
    def __init__(self, annotation_lines, input_shape, batch_size, num_classes, mode, dataset_path, pseudo_label_confidence_file='splits/id_to_reliability_sorted.txt'):
        self.annotation_lines = annotation_lines
        self.length = len(self.annotation_lines)
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.mode = mode
        self.dataset_path = dataset_path
        self.epoch = 0
        self.total_epochs = 150  # the total epoch
        self.current_epoch_batch_indices = []

        self.weight_save_dir = 'sample_weights'
        self.weight_save_path = os.path.join(self.weight_save_dir, 'sample_weights.txt')


        if not os.path.exists(self.weight_save_dir):
            os.makedirs(self.weight_save_dir)


        self.pseudo_confidence = self.read_confidence_file(pseudo_label_confidence_file)

        self.real_image_weight = 1.0
        self.synthetic_image_weight = 1.0

        self.annotation_lines, self. sample_weights = self.oversample_real_images(self.annotation_lines)


    def read_confidence_file(self, pseudo_label_confidence_file):
        if pseudo_label_confidence_file is None:
            return None
        with open(pseudo_label_confidence_file, 'r') as f:
            lines = f.readlines()
        confidence_dict = {line.split()[0]: float(line.split()[1]) for line in lines}
        return confidence_dict

    def oversample_real_images(self, annotation_lines):
        """The sampling weights were adaptively adjusted based on image categories"""

        real_image_lines = []
        sample_weights = []
        real_image_count = 207
        synthetic_image_count = 500

        repeat_factor = math.ceil(synthetic_image_count / real_image_count)

        #  over-sample
        for line in annotation_lines:
            img_id = line.split()[0]
            img_num = int(img_id.split(".")[0])
            if img_num >= 1 and img_num <= 256:
                real_image_lines.extend([line] * repeat_factor)
                sample_weights.extend([self.real_image_weight] * repeat_factor)

            else:
                # synthetic images
                real_image_lines.append(line)
                sample_weights.append(self.get_synthetic_image_weight(img_id))
        return real_image_lines, sample_weights

    def get_synthetic_image_weight(self, img_id):
        """Initial weights of the synthesized images were adjusted based on their pseudo‐label confidence scores"""
        if self.pseudo_confidence is None or img_id not in self.pseudo_confidence:
            return self.synthetic_image_weight

        confidence = self.pseudo_confidence[img_id]

        # Images with high confidence scores were assigned higher weights, while those with low confidence scores were assigned lower weights.
        adjusted_weight = self.synthetic_image_weight * (confidence ** 2)

        return adjusted_weight

    def update_sampling_weights(self, epoch, total_epochs):
        """
        Sampling weights are dynamically updated at the end of each epoch to progressively incorporate lower-confidence pseudo-labels.

        """
        confidence_scale_factor = min(1.0, epoch / total_epochs)

        self.synthetic_image_weight = 1.0 + confidence_scale_factor * 0.05
        self.real_image_weight = 1.0


        self.annotation_lines, self.sample_weights = self.oversample_real_images(self.annotation_lines)

        self.save_sample_weights(epoch)


    def save_sample_weights(self, epoch):
        if epoch not in [0, 30, 60, 90, 120, 150]:
            return


        if not os.path.exists(self.weight_save_dir):
            os.makedirs(self.weight_save_dir)

        weight_file_name = f'ep{epoch}.txt'
        weight_save_path = os.path.join(self.weight_save_dir, weight_file_name)

        with open(weight_save_path, 'w') as f:
            for i, batch_indices in enumerate(self.current_epoch_batch_indices):
                f.write(f"Batch {i + 1}\n")
                for idx in batch_indices:
                    img_id = self.annotation_lines[idx].split()[0]
                    weight = self.sample_weights[idx]
                    f.write(f"{img_id} {weight:.4f}\n")
                f.write("------------------\n")
        print(f"Sample weights saved to {weight_save_path}")

    def __len__(self):
        return math.ceil(len(self.annotation_lines) / float(self.batch_size))

    def __getitem__(self, index):
        images = []
        targets = []

        sampled_indices = np.random.choice(np.arange(len(self.annotation_lines)),
                                           size=self.batch_size,
                                           replace=True,
                                           p=self.sample_weights / np.sum(self.sample_weights))


        self.current_epoch_batch_indices.append(sampled_indices)

        for i in sampled_indices:
            id = self.annotation_lines[i].split()[0]

            img = Image.open(os.path.join(self.dataset_path, "VOC2007/JPEGImages", id + ".jpg"))
            mask = Image.open(os.path.join(self.dataset_path, "VOC2007/SegmentationClass", id + ".png"))

            img, mask = self.get_random_data(img, mask, self.input_shape, id)
            img = preprocess_input(np.array(img, np.float64))

            mask = np.array(mask)
            mask[mask >= self.num_classes] = self.num_classes

            seg_labels = np.eye(self.num_classes)[mask.reshape([-1])]
            seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes))

            images.append(img)
            targets.append(seg_labels)

        images = np.array(images)
        targets = np.array(targets)
        return images, targets

    def on_epoch_end(self):
        """
        Weight Update Logic: Upon completion of the specified epoch,
        the sampling weights for the synthesized images are dynamically adjusted using the update_sampling_weights() function.
        """
        # Called every n epochs to update the sampling weights and shuffle the data.
        n = 30  #
        if self.epoch % n == 0:
            print('Updating sampling weights.')

            self.save_sample_weights(self.epoch)
            self.update_sampling_weights(self.epoch, self.total_epochs)

        shuffle(self.annotation_lines)

        self.current_epoch_batch_indices = []

        self.epoch += 1

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, img, mask, input_shape, id, jitter=.3):

        if self.mode == 'train':
            img = gray2rgb(img)
            mask = Image.fromarray(np.array(mask))

            iw, ih = img.size
            h, w = input_shape

            # Random Scale
            new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
            scale = self.rand(0.25, 2)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            img = img.resize((nw, nh), Image.BICUBIC)
            mask = mask.resize((nw, nh), Image.NEAREST)

            # Random flip
            flip = self.rand() < .5
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

            # Rando Crop
            dx = int(self.rand(0, w - nw))
            dy = int(self.rand(0, h - nh))
            new_img = Image.new('RGB', (w, h), (128, 128, 128))
            new_mask = Image.new('L', (w, h), (0))
            new_img.paste(img, (dx, dy))
            new_mask.paste(mask, (dx, dy))
            img = new_img
            mask = new_mask

            img = np.array(img, np.uint8)
            return img, mask

        if self.mode == 'val':
            img = gray2rgb(img)
            mask = Image.fromarray(np.array(mask))
            return img, mask
#


