import colorsys
import copy
import time
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
import PIL
from matplotlib.path import Path
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
color_list=["#f79256","#fbd1a2","#7dcfb6","#00b2ca"]

from model.unet import ConvNeXt_Unet as unet
from utils import gray2rgb, preprocess_input, resize_image, show_config
np.set_printoptions(suppress=True)
from tta_wrapper import tta_segmentation
import os
np.set_printoptions(threshold=np.inf)
import timeit
import torch

class Unet(object):
    _defaults = {

        "model_path": 'best.h5',
        # ---------------------------------------------
        "num_classes"       : 4,
        # ---------------------------------------------
        "backbone"          : "ConvNeXtTiny",
        # ---------------------------------------------
        "input_shape"       : [256, 256],
        # ---------------------------------------------
        "mix_type"          :1,
    }

    #---------------------------------------------------#
    #   UNET
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        #---------------------------------------------------#
        #   color setting
        #---------------------------------------------------#

        self.colors = [(244, 202, 52), (116, 182, 84), (87, 114, 202), (250, 23, 24)]

        #---------------------------------------------------#
        #   generate model
        #---------------------------------------------------#
        self.generate()

        show_config(**self._defaults)
        self.background = 0
        self.microfracture = 0
        self.organic = 0
        self.inorganic = 0

    #---------------------------------------------------#
    #   load model
    #---------------------------------------------------#
    def generate(self):
        #-------------------------------#
        #   load model
        #-------------------------------#
        self.model = unet()
        self.model.load_weights(self.model_path)

    def filter_pseudo_labels(self, pr):
        pred_u = torch.from_numpy(pr).unsqueeze(0).permute(0, 3, 1, 2)  # [1,4,256,256]

        logits_u_aug, label_u_aug = torch.max(pred_u, dim=1)  # [1,256,256]

        # obtain confidence
        entropy = -torch.sum(pred_u * torch.log(pred_u + 1e-10), dim=1)
        entropy /= np.log(4)  # torch.Size([1, 256, 256])

        confidence = 1.0 - entropy

        # The average confidence for each image was computed by averaging the confidence values across the height (H) and width (W) dimensions.
        mean_confidence = confidence.mean(dim=[1, 2])

        confidence = confidence * logits_u_aug
        confidence = confidence.mean(dim=[1, 2])  # 1*C
        confidence = confidence.cpu().numpy().tolist()

        return confidence

    #---------------------------------------------------#
    #   inference
    #---------------------------------------------------#
    def detect_image(self, image, count=False, name_classes=['pore','clay','quartz','pyrite'],return_mask=True):

        image       = gray2rgb(image)

        old_img     = copy.deepcopy(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]

        # -------------------------------------------------------------------------------------
        image_data, nw, nh  = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        # -------------------------------------------------------------------------------------
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)

        pr = self.model.predict(image_data)[0]


        pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)

        confidence = self.filter_pseudo_labels(pr)
        pr = pr.argmax(axis=-1)

        binary_image = pr

        #---------------------------------------------------------#
        #   count pixel number
        #---------------------------------------------------------#
        if count:
            classes_nums        = np.zeros([self.num_classes])
            total_points_num    = orininal_h * orininal_w
            ratio_list = []
            print('-' * 63)
            print("|%25s | %15s | %15s|"%("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(4):
                num     = np.sum(pr == i)
                ratio   = num / total_points_num * 100
                if num > 0:
                    print("|%15s | %15s | %14.2f%%|"%(str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
                ratio_list.append(float(ratio))
            # print("classes_nums:", classes_nums)

        if self.mix_type == 0:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            image   = Image.fromarray(np.uint8(seg_img))
            image   = Image.blend(old_img, image, 0.6)
        #
        elif self.mix_type == 1:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            image   = Image.fromarray(np.uint8(seg_img))

        # elif self.mix_type == 2:
        #     seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
        #     image = Image.fromarray(np.uint8(seg_img))

        #return image,ratio_list
        if return_mask:
            return image, binary_image,confidence

        return image

    def get_miou_png(self, image):

        image       = gray2rgb(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]

        image_data, nw, nh  = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)

        pr = self.model.predict(image_data)[0]

        pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]

        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
        pred = pr.argmax(axis=-1)

        image = Image.fromarray(np.uint8(pred))
        return image



if __name__ == "__main__":
    unet  = Unet()

    mode = "predict"
    count = False


    name_classes=['pore','clay','quartz','pyrite']

    if mode == "predict":

        id_to_reliability = []

        for i, filename in enumerate(os.listdir('./synthetic/207-fid55')):
               image = Image.open('./synthetic/207-fid55/'+ filename)

               r_image, mask, confidence = unet.detect_image(image, count=count, name_classes=name_classes)


               cv2.imwrite(r'synthetic'+ '\mask' + f'\{str(i + 257).zfill(6)}.png', mask)


               print(f'\n================> Index{i} : Finish predict, confidence:{confidence}')

               id_to_reliability.append((filename.split('.jpg')[0], np.round(confidence[0], 5)))

        print('=' * 50)

        id_to_reliability.sort(key=lambda elem: elem[1], reverse=True)

        with open(os.path.join('splits', 'id_to_reliability_sorted.txt'), 'w') as f:
             for elem in id_to_reliability:
                f.write(f'{elem[0]} {elem[1]:.5f}\n')

        print('Sorted reliability data written to id_to_reliability_sorted.txt')

