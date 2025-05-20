import tensorflow as tf
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
from PIL import Image
from tensorflow.python.keras.layers import VersionAwareLayers
layers = VersionAwareLayers()
import scipy.signal
import csv
import shutil
import numpy as np
import random
import math
import cv2
import os

from os.path import join
import matplotlib.pyplot as plt
from tensorflow.keras import backend
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tqdm import tqdm

import time
import visualkeras
import h5py
import colorsys
import copy
from libtiff import TIFF
from scipy import misc
import os.path as osp
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import torch.nn as nn



def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)



# ------------------------------------------------------
class Evaluator(object):
    def __init__(self, num_class, ignore=False):
        self.num_class = num_class
        self.ignore = ignore  # whether to consider ignore class, True when evaluate seed quality
        self.confusion_matrix = np.zeros(shape=(self.num_class, self.num_class))  # index 0: gt / index 1: pred

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Precision_Recall(self):
        precision = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=0) + 1e-5)
        recall = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1) + 1e-5)
        if self.ignore:
            mp = np.nanmean(precision[:-1])
            mr = np.nanmean(recall[:-1])
            return precision[:-1], recall[:-1], mp, mr
        else:
            mp = np.nanmean(precision)
            mr = np.nanmean(recall)
            return precision, recall, mp, mr

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        IoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        if self.ignore:
            MIoU = np.nanmean(IoU[:-1])
            return IoU[:-1], MIoU
        else:
            MIoU = np.nanmean(IoU)
            return IoU, MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape, "gt: {} pred: {}".format(gt_image.shape, pre_image.shape)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


# <=============================== Freeze BN ================================>
def freeze_model(model, freeze_batch_norm=False):
    '''
    freeze a keras model

    Input
    ----------
        model: a keras model
        freeze_batch_norm: False for not freezing batch notmalization layers
    '''
    if freeze_batch_norm:
        for layer in model.layers:
            layer.trainable = False
    else:
        from tensorflow.keras.layers import BatchNormalization
        for layer in model.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False
    return model


# <=============================== Gray2RGB ================================>
def gray2rgb(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 


# <============================== Preprocess ===============================>
def preprocess_input(image, type='convnext'): # type='convnext'
    if type != 'convnext':
        image = image / 127.5 - 1
    else:
        image = layers.Normalization( mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                      variance=[(0.229 * 255) ** 2,
                                                (0.224 * 255) ** 2,
                                                (0.225 * 255) ** 2])(image)

    return image

# <================================ Resize =================================>
def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))


    return new_image, nw, nh

# <============================== Show Config ===============================>
def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    logger.info('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    logger.info('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    logger.info('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
        logger.info('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)
    logger.info('-' * 70 + '\n')


# <============================== Count Flops ===============================>
def try_count_flops(model: Union[tf.Module, tf.keras.Model],
                    inputs_kwargs: Optional[Dict[str, Any]] = None,
                    output_path: Optional[str] = None):
  """Counts and returns model FLOPs.
  Args:
    model: A model instance.
    inputs_kwargs: An optional dictionary of argument pairs specifying inputs'
      shape specifications to getting corresponding concrete function.
    output_path: A file path to write the profiling results to.
  Returns:
    The model's FLOPs.
  """
  if hasattr(model, 'inputs'):
    try:
      if model.inputs:
        inputs = [
            tf.TensorSpec([1] + input.shape[1:], input.dtype)
            for input in model.inputs
        ]
        concrete_func = tf.function(model).get_concrete_function(inputs)
      else:
        concrete_func = tf.function(model.call).get_concrete_function(
            **inputs_kwargs)
      frozen_func, _ = convert_variables_to_constants_v2_as_graph(concrete_func)

      # Calculate FLOPs.
      run_meta = tf.compat.v1.RunMetadata()
      opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
      if output_path is not None:
        opts['output'] = f'file:outfile={output_path}'
      else:
        opts['output'] = 'none'
      flops = tf.compat.v1.profiler.profile(
          graph=frozen_func.graph, run_meta=run_meta, options=opts)
      return flops.total_float_ops
    except Exception as e:
      logging.info(
          'Failed to count model FLOPs with error %s, because the build() '
          'methods in keras layers were not called. This is probably because '
          'the model was not feed any input, e.g., the max train step already '
          'reached before this run.', e)
      return None
  return None

# <============================== Count Size ===============================>
def get_model_size(model):
    """
       Return model size(MB)
    """
    param_num = sum([np.prod(w.shape) for w in model.get_weights()])
    param_size = param_num * 4 / 1024 / 1024
    return param_size


# <============================== Plot Model ===============================>
def plot_model(model):
    """
       Plot structure of the given model and save the corresponding figure.
    """

    return tf.keras.utils.plot_model(model, to_file='model.png',
                                     show_shapes=True,
                                     show_layer_names=True)


# <============================== Layer View ===============================>
def layer_view(model):
    """
       Visualize every layer and plot it with Keras API.
    """
    return visualkeras.layered_view(model,legend=True, draw_volume=True).show()


# <============================== Training Log ===============================>
logs = set()
def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"

    logger = logging.getLogger(name)
    logger.setLevel(level)

    ch = logging.FileHandler(name, 'w')
    ch.setFormatter(format_str)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0

    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

logger = init_log('training-logs.log')



class meanIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))

        return iu, np.nanmean(iu)



#----------------------------------------------------------------------------
#  Useful Callbacks
class LossHistory(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, val_loss_flag=True):
        self.log_dir    = log_dir
        self.losses     = []
        self.val_loss_flag  = val_loss_flag

        self.losses         = []
        if self.val_loss_flag:
            self.val_loss   = []
        
        os.makedirs(self.log_dir)

    def on_epoch_end(self, epoch, logs={}):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(logs.get('loss'))
        if self.val_loss_flag:
            self.val_loss.append(logs.get('val_loss'))
        
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(logs.get('loss')))
            f.write("\n")
        if self.val_loss_flag:
            with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
                f.write(str(logs.get('val_loss')))
                f.write("\n")
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        if self.val_loss_flag:
            plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
            
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            if self.val_loss_flag:
                plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('A Loss Curve')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

class ExponentDecayScheduler(tf.keras.callbacks.Callback):
    def __init__(self,
                 decay_rate,
                 verbose=0):
        super(ExponentDecayScheduler, self).__init__()
        self.decay_rate         = decay_rate
        self.verbose            = verbose
        self.learning_rates     = []

    def on_epoch_end(self, batch, logs=None):
        learning_rate = K.get_value(self.model.optimizer.lr) * self.decay_rate
        K.set_value(self.model.optimizer.lr, learning_rate)
        if self.verbose > 0:
            print('Setting learning rate to %s.' % (learning_rate))

class WarmUpCosineDecayScheduler(tf.keras.callbacks.Callback):
    def __init__(self, T_max, eta_min=0, verbose=0):
        super(WarmUpCosineDecayScheduler, self).__init__()
        self.T_max      = T_max
        self.eta_min    = eta_min
        self.verbose    = verbose
        self.init_lr    = 0
        self.last_epoch = 0

    def on_train_begin(self, batch, logs=None):
        self.init_lr = K.get_value(self.model.optimizer.lr)

    def on_epoch_end(self, batch, logs=None):
        learning_rate = self.eta_min + (self.init_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
        self.last_epoch += 1

        K.set_value(self.model.optimizer.lr, learning_rate)
        if self.verbose > 0:
            print('Setting learning rate to %s.' % (learning_rate))
    
class ParallelModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only,mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint,self).set_model(self.single_model)


class DisplayCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_body ,input_shape, image_ids ,dataset_path ,epoch_interval=1):
        self.epoch_interval = epoch_interval
        self.model_body = model_body
        self.input_shape = input_shape
        self.image_ids = image_ids
        self.dataset_path = dataset_path
        self.colors = [(247, 146, 86), (251, 209, 162), (125, 207, 182), (0, 178, 202)]

    def on_epoch_end(self, epoch, logs=None):
            # ------------------------------
            #  images from validation set for visualization
            # ------------------------------
            random_index = random.randint(0, 68)
            random_image_id = self.image_ids[random_index].split()[0]

            #-------------------------------
            #   load images from validation set
            #-------------------------------
            image_path  = os.path.join(self.dataset_path, "VOC2007/JPEGImages/"+ random_image_id +".jpg")
            image       = Image.open(image_path)

            # -------------------------------
            #   load mask
            # -------------------------------
            mask_path = os.path.join(self.dataset_path, "VOC2007/SegmentationClass/"+ random_image_id +".png")
            mask = np.array(Image.open(mask_path))

            # -------------------------------
            #   images preprocessing
            # -------------------------------
            orininal_h = np.array(image).shape[0]
            orininal_w = np.array(image).shape[1]
            image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
            image_data = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)

            # -------------------------------
            #   forward process
            # -------------------------------
            pr = self.model_body.predict(image_data)[0]

            # ---------------------------------------------------
            #   The gray portion of the image was cropped out
            # ---------------------------------------------------
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
            int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

            # ---------------------------------------------------
            #   resize the image
            # ---------------------------------------------------
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)

            # ---------------------------------------------------
            #   The class of each pixel was extracted
            # ---------------------------------------------------
            pr = pr.argmax(axis=-1)

            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            mask_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(mask, [-1])], [orininal_h, orininal_w, -1])
            true_mask = Image.fromarray(np.uint8(mask_img))
            pred_mask = Image.fromarray(np.uint8(seg_img))

            # ------------------------------------------------
            #  visualize original image with its mask and the final prediction
            # ------------------------------------------------
            cmap = ["summer",'rainbow']
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
            ax[0].imshow(image,cmap=cmap[1])
            ax[0].tick_params(direction='out',length=1.)
            x_label = ax[0].get_xticklabels()
            [x_label_custom.set_fontname('serif') for x_label_custom in x_label]
            y_label = ax[0].get_yticklabels()
            [y_label_custom.set_fontname('serif') for y_label_custom in y_label]
            ax[0].set_title(f"Image: {epoch:03d}",fontname='serif')

            ax[1].imshow(true_mask)
            ax[1].tick_params(direction='out',length=1.)
            x_label = ax[1].get_xticklabels()
            [x_label_custom.set_fontname('serif') for x_label_custom in x_label]
            y_label = ax[1].get_yticklabels()
            [y_label_custom.set_fontname('serif') for y_label_custom in y_label]
            ax[1].set_title(f"Ground Truth Mask: {epoch:03d}",fontname='serif')

            ax[2].imshow(pred_mask)
            ax[2].tick_params(direction='out',length=1.)
            x_label = ax[2].get_xticklabels()
            [x_label_custom.set_fontname('serif') for x_label_custom in x_label]
            y_label = ax[2].get_yticklabels()
            [y_label_custom.set_fontname('serif') for y_label_custom in y_label]
            ax[2].set_title(f"Predicted Mask: {epoch:03d}",fontname='serif')

            plt.show()
            plt.close()

class EvalCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_body, input_shape, num_classes, image_ids, dataset_path, log_dir,\
            miou_out_path=".temp_miou_out", eval_flag=True, period=1):
        super(EvalCallback, self).__init__()
        
        self.model_body         = model_body
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.image_ids          = image_ids
        self.dataset_path       = dataset_path
        self.log_dir            = log_dir
        self.miou_out_path      = miou_out_path
        self.eval_flag          = eval_flag
        self.period             = period
        
        self.image_ids          = [image_id.split()[0] for image_id in image_ids]
        self.mious      = [0]
        self.epoches    = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def get_miou_png(self, image):

        image       = gray2rgb(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]

        image_data, nw, nh  = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)

        pr = self.model_body.predict(image_data)[0]
        pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]

        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)

        pr = pr.argmax(axis=-1)

        image = Image.fromarray(np.uint8(pr))
        return image
    
    def on_epoch_end(self, epoch, logs=None):
        temp_epoch = epoch + 1
        if temp_epoch % self.period == 0 and self.eval_flag:
            gt_dir      = os.path.join(self.dataset_path, "VOC2007/SegmentationClass/")
            pred_dir    = os.path.join(self.miou_out_path, 'detection-results')
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)

            tbar = tqdm(self.image_ids)
            for image_id in tbar:

                image_path  = os.path.join(self.dataset_path, "VOC2007/JPEGImages/"+image_id+".jpg")
                image       = Image.open(image_path)

                image       = self.get_miou_png(image)
                image.save(os.path.join(pred_dir, image_id + ".png"))
                tbar.set_description("- Seg - INFO - Summary")

            _, IoUs, _, _ ,F1_score= compute_mIoU(gt_dir, pred_dir, self.image_ids, self.num_classes, None)  # 执行计算mIoU的函数
            temp_miou = np.nanmean(IoUs) * 100
            
            self.mious.append(temp_miou)
            self.epoches.append(temp_epoch)

            if temp_epoch < 50 or 50 < temp_miou <= 150:
               logger.info('===========> Epoch: {:}, MeanIoU: {:.2f}'.format(
                temp_epoch, temp_miou))
            else:
                logger.info('===========> Epoch: {:}, MeanIoU: {:.2f}\n'.format(
                    temp_epoch, temp_miou))

            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(temp_miou))
                f.write("\n")
            
            plt.figure()
            plt.plot(self.epoches, self.mious, 'red', linewidth = 2, label='train miou')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Miou')
            plt.title('A Miou Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
            plt.cla()
            plt.close("all")

            shutil.rmtree(self.miou_out_path)


#----------------------------------------------------------------------------
# Useful metrics
def Iou_score(smooth = 1e-5, threhold = 0.5):
    def _Iou_score(y_true, y_pred):
        # score calculation
        y_pred = backend.greater(y_pred, threhold)
        y_pred = backend.cast(y_pred, backend.floatx())
        intersection = backend.sum(y_true * y_pred, axis=[0,1,2])
        union = backend.sum(y_true + y_pred, axis=[0,1,2]) - intersection

        score = (intersection + smooth) / (union + smooth)
        return score
    return _Iou_score

def f_score(beta=1, smooth = 1e-5, threhold = 0.5):
    def f1_score(y_true, y_pred):
        y_pred = backend.greater(y_pred, threhold)
        y_pred = backend.cast(y_pred, backend.floatx())

        tp = backend.sum(y_true * y_pred, axis=[0,1,2])
        fp = backend.sum(y_pred         , axis=[0,1,2]) - tp
        fn = backend.sum(y_true         , axis=[0,1,2]) - tp

        score = ((1 + beta ** 2) * tp + smooth) \
                / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        return score
    return f1_score

# 设标签宽W，长H
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 

def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1) 

def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1) 

def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None):  

    hist = np.zeros((num_classes, num_classes))

    gt_imgs     = [join(gt_dir, x + ".png") for x in png_name_list]  
    pred_imgs   = [join(pred_dir, x + ".png") for x in png_name_list]  

    for ind in range(len(gt_imgs)): 

        pred = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))  


        if len(label.flatten()) != len(pred.flatten()):  
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        if name_classes is not None and ind > 0 and ind % 10 == 0: 
            print('{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; Accuracy-{:0.2f}%'.format(
                    ind, 
                    len(gt_imgs),
                    100 * np.nanmean(per_class_iu(hist)),
                    100 * np.nanmean(per_class_PA_Recall(hist)),
                    100 * per_Accuracy(hist)
                )
            )
    IoUs        = per_class_iu(hist)
    PA_Recall   = per_class_PA_Recall(hist)
    Precision   = per_class_Precision(hist)

    if name_classes is not None:
        for ind_class in range(num_classes):
            print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
                + '; Recall (equal to the PA)-' + str(round(PA_Recall[ind_class] * 100, 2))+ '; Precision-' + str(round(Precision[ind_class] * 100, 2)))



    from prettytable import PrettyTable
    field_names = ('mIoU', 'mAcc', 'aAcc')
    table = PrettyTable(field_names=field_names)
    table.add_row(["%.2f" % round(np.nanmean(IoUs) * 100, 2) , "%.2f" % round(np.nanmean(PA_Recall) * 100, 2), "%.2f" % round(per_Accuracy(hist) * 100, 2)])
    print(table)

    return np.array(hist, np.int), IoUs, PA_Recall, Precision, 2*PA_Recall*Precision/(PA_Recall+Precision)

def adjust_axes(r, t, fig, axes):
    bb                  = t.get_window_extent(renderer=r)
    text_width_inches   = bb.width / fig.dpi
    current_fig_width   = fig.get_figwidth()
    new_fig_width       = current_fig_width + text_width_inches
    propotion           = new_fig_width / current_fig_width
    x_lim               = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])

def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size = 20, plt_show = True):
    fig     = plt.gcf() 
    axes    = plt.gca()
    plt.barh(range(len(values)), values, height=0.65, color='#8dbad9')

    plt.title(plot_title, fontsize=tick_font_size + 2, fontname='serif')

    plt.xlabel(x_label, fontsize=tick_font_size, fontname='serif')
    plt.xticks(fontname='serif', size=15)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size +2, fontname='serif')

    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val) 
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color='black', va='center', fontweight='bold', fontname='serif')
        if i == (len(values)-1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()

def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, F1_score, name_classes, tick_font_size = 12):
    draw_plot_func(IoUs, name_classes, "mIoU = {0:.2f}%".format(np.nanmean(IoUs)*100), "Intersection over Union", \
        os.path.join(miou_out_path, "mIoU.png"), tick_font_size = tick_font_size, plt_show = True)


    draw_plot_func(PA_Recall, name_classes, "mPA = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Pixel Accuracy", \
        os.path.join(miou_out_path, "mPA.png"), tick_font_size = tick_font_size, plt_show = False)

    
    draw_plot_func(PA_Recall, name_classes, "mRecall = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Recall", \
        os.path.join(miou_out_path, "Recall.png"), tick_font_size = tick_font_size, plt_show = False)


    draw_plot_func(Precision, name_classes, "mPrecision = {0:.2f}%".format(np.nanmean(Precision)*100), "Precision", \
        os.path.join(miou_out_path, "Precision.png"), tick_font_size = tick_font_size, plt_show = False)


    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer          = csv.writer(f)
        writer_list     = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)

    print(f"F1_score : {np.sum(F1_score) * 0.25:.4f}")

    mean_precision = np.nanmean(Precision) * 100
    print(f"Precision : {mean_precision:.2f}%")

    mean_recall = np.nanmean(PA_Recall) * 100
    print(f"Recall : {mean_recall:.2f}%")

    pixel_accuracy = per_Accuracy(hist) * 100
    print(f"PixelAccuracy : {pixel_accuracy:.2f}%")



#----------------------------------------------------------------------------