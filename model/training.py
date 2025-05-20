import math
from functools import partial
from tensorflow.keras.losses import binary_crossentropy
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Flatten
from tensorflow.keras import layers
from scipy.ndimage import distance_transform_edt as distance

from tensorflow.keras import losses
import itertools
from typing import Any, Optional

_EPSILON = tf.keras.backend.epsilon()

def pixelwise_l2_loss(y_true, y_pred):
    y_true /= 255.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return K.mean(K.square(y_true_f - y_pred_f))

@tf.keras.utils.register_keras_serializable()
class BoundaryLoss(tf.keras.losses.Loss):
    def __init__(self, from_logits: bool = False, use_multi_gpu: bool = False,
                 global_batch_size: int = 16, num_classes: int = 1,
                 boundary_alpha: float = 20., **kwargs):
        """
        Args:
            BoundaryLoss is the sum of semantic segmentation loss.
            The BoundaryLoss loss is a binary cross entropy loss.

            from_logits       (bool)  : When softmax is not applied to the activation
                                        layer of the last layer of the model.
            use_multi_gpu     (bool)  : To calculate the loss for each gpu when using distributed training.
            global_batch_size (int)   : Global batch size (Batch_size = GLOBAL_BATCH_SIZE / GPUs)
            num_classes       (int)   : Number of classes to classify (must be equal to number of last filters in the model)
            boundary_alpha    (float) : Boundary loss alpha
        """
        super().__init__(**kwargs)
        self.from_logits = from_logits
        self.use_multi_gpu = use_multi_gpu
        self.global_batch_size = global_batch_size
        self.num_classes = num_classes
        self.boundary_alpha = boundary_alpha

        if self.use_multi_gpu:
            self.loss_reduction = losses.Reduction.NONE
        else:
            self.loss_reduction = losses.Reduction.AUTO

    def get_config(self):
        config = super().get_config()
        config.update(from_logits=self.from_logits, use_multi_gpu=self.use_multi_gpu,
                      global_batch_size=self.global_batch_size, num_classes=self.num_classes)
        return config

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        # Calc bce loss
        edge_map = tf.cast(y_true, dtype=tf.float32)
        grad_components = tf.image.sobel_edges(edge_map)
        grad_mag_components = grad_components ** 2

        grad_mag_square = tf.math.reduce_sum(grad_mag_components, axis=-1)

        edge = tf.sqrt(grad_mag_square)
        edge = tf.cast(tf.where(edge != 0, 1., 0.), dtype=tf.float32)

        edge = tf.reshape(edge, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        pos_index = (edge == 1.)
        neg_index = (edge == 0.)

        weight = tf.zeros_like(edge, dtype=tf.float32)

        pos_num = tf.reduce_sum(tf.cast(pos_index, dtype=tf.float32))
        neg_num = tf.reduce_sum(tf.cast(neg_index, dtype=tf.float32))

        sum_num = pos_num + neg_num

        weight = tf.where(pos_index, neg_num * 1.0 / sum_num, pos_num * 1.0 / sum_num)

        loss = tf.nn.weighted_cross_entropy_with_logits(labels=edge, logits=y_pred, pos_weight=weight)

        # Reduce loss to scalar
        if self.use_multi_gpu:
            loss = tf.reduce_mean(loss)
        else:
            loss = tf.reduce_mean(loss)

        loss *= self.boundary_alpha
        return loss


@tf.keras.utils.register_keras_serializable()
class AuxiliaryLoss(tf.keras.losses.Loss):
    def __init__(self, from_logits: bool = False, use_multi_gpu: bool = False,
                 global_batch_size: int = 16, num_classes: int = 1,
                 aux_alpha: float = 0.4,
                 **kwargs):
        """
        Args:
            AuxiliaryLoss is the sum of semantic segmentation loss.
            The AuxiliaryLoss loss is a cross entropy loss.

            from_logits       (bool)  : When softmax is not applied to the activation
                                        layer of the last layer of the model.
            use_multi_gpu     (bool)  : To calculate the loss for each gpu when using distributed training.
            global_batch_size (int)   : Global batch size (Batch_size = GLOBAL_BATCH_SIZE / GPUs)
            num_classes       (int)   : Number of classes to classify (must be equal to number of last filters in the model)
            aux_alpha         (float) : Aux loss alpha.
        """
        super().__init__(**kwargs)
        self.from_logits = from_logits
        self.use_multi_gpu = use_multi_gpu
        self.global_batch_size = global_batch_size
        self.num_classes = num_classes
        self.aux_alpha = aux_alpha

        if self.use_multi_gpu:
            self.loss_reduction = losses.Reduction.NONE
        else:
            self.loss_reduction = losses.Reduction.AUTO

    def get_config(self):
        config = super().get_config()
        config.update(from_logits=self.from_logits, use_multi_gpu=self.use_multi_gpu,
                      global_batch_size=self.global_batch_size, num_classes=self.num_classes)
        return config

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        loss = losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=self.loss_reduction)(y_true=y_true, y_pred=y_pred)

        if self.use_multi_gpu:
            loss = tf.reduce_mean(loss)

        loss *= self.aux_alpha
        return loss



def segmentation_boundary_loss(y_true, y_pred, axis = (1, 2), smooth = 1e-5):
    """
    Paper Implemented : https://arxiv.org/abs/1905.07852
    Using Binary Segmentation mask, generates boundary mask on fly and claculates boundary loss.
    :param y_true:
    :param y_pred:
    :return:
    """
    InvPred = 1 - y_pred
    InvTrue = 1 - y_true
    y_pred_bd = tf.nn.max_pool2d(InvPred, (3, 3), (1, 1), padding = 'SAME' )
    y_true_bd = tf.nn.max_pool2d(InvTrue, (3, 3), (1, 1), padding = 'SAME' )

    y_pred_bd = y_pred_bd - InvPred
    y_true_bd = y_true_bd - InvTrue

    y_pred_bd_ext = tf.nn.max_pool2d(InvPred, (5, 5), (1, 1), padding = 'SAME' )
    y_true_bd_ext = tf.nn.max_pool2d(InvTrue, (5, 5), (1, 1), padding = 'SAME' )

    y_pred_bd_ext = y_pred_bd_ext - InvPred
    y_true_bd_ext = y_true_bd_ext - InvTrue

    P = tf.reduce_sum(y_pred_bd * y_true_bd_ext, axis = axis) / (tf.reduce_sum(y_pred_bd, axis = axis) + smooth)
    R = tf.reduce_sum(y_true_bd * y_pred_bd_ext, axis = axis) / (tf.reduce_sum(y_true_bd, axis = axis) + smooth)

    F1_Score = (2 * P * R) / (P + R + smooth)
    loss = 1. - F1_Score
    return loss


def calc_dist_map(seg):
    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

    return res


def calc_dist_map_batch(y_true):
    y_true_numpy = y_true.numpy()
    return np.array([calc_dist_map(y)
                     for y in y_true_numpy]).reshape(y_true.shape).astype(np.float32)


def boundary_loss(y_true, y_pred):
    y_true_dist_map = tf.py_function(func=calc_dist_map_batch,
                                     inp=[y_true],
                                     Tout=tf.float32)
    multipled = y_pred * y_true_dist_map
    return K.mean(multipled)


def dice_loss_with_CE(cls_weights, beta=1, smooth = 1e-5):
    """Sum of cross entropy and dice losses:

       .. math:: L(A, B) = bce_weight * crossentropy(A, B) + dice_loss(A, B)

       Args:
           gt: ground truth 4D keras tensor (B, H, W, C)
           pr: prediction 4D keras tensor (B, H, W, C)
           class_weights: 1. or list of class weights for dice loss, len(weights) = C
           smooth: value to avoid division by zero
           per_image: if ``True``, dice loss is calculated as mean over images in batch (B),
               else over whole batch
           beta: coefficient for precision recall balance

       Returns:
           loss

    """
    cls_weights = np.reshape(cls_weights, [1, 1, 1, -1])
    def _dice_loss_with_CE(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        CE_loss = - y_true * K.log(y_pred) * cls_weights
        CE_loss = K.mean(K.sum(CE_loss, axis = -1))

        tp = K.sum(y_true * y_pred, axis=[0,1,2])
        fp = K.sum(y_pred         , axis=[0,1,2]) - tp
        fn = K.sum(y_true         , axis=[0,1,2]) - tp

        score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        score = tf.reduce_mean(score)
        dice_loss = 1 - score

        return CE_loss + dice_loss
    return _dice_loss_with_CE

def CE(cls_weights):
    """
       Multi-class weighted cross entropy.

       WCE(p, p̂) = −Σp*log(p̂)*class_weights

       Used as loss function for multi-class image segmentation with one-hot encoded masks.
       :param class_weights: Weight coefficients (list of floats)
       :return: Weighted cross entropy loss function (Callable[[tf.Tensor, tf.Tensor], tf.Tensor])
    """
    cls_weights = np.reshape(cls_weights, [1, 1, 1, -1])
    def _CE(y_true, y_pred):
        """
                Computes the weighted cross entropy.

                :param y_true: Ground truth (tf.Tensor, shape=(None, None, None, None))
                :param y_pred: Predictions (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
                :return: Weighted cross entropy (tf.Tensor, shape=(<BATCH_SIZE>,))
        """
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        # p * log(p̂) * class_weights
        # y_true:[1,256,256,4] | y_pred:[1,256,256,4] -> [1,256,256,4]
        CE_loss = - y_true * K.log(y_pred) * cls_weights
        # (1, 256, 256) ->  return scalar
        CE_loss = K.mean(K.sum(CE_loss, axis = -1))

        return CE_loss
    return _CE



def dice_loss_with_Focal_Loss(cls_weights, beta=1, smooth = 1e-5, alpha=0.5, gamma=2):
    cls_weights = np.reshape(cls_weights, [1, 1, 1, -1])
    def _dice_loss_with_Focal_Loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        logpt = - y_true * K.log(y_pred) * cls_weights
        logpt = - K.sum(logpt, axis = -1)

        pt = tf.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        CE_loss = -((1 - pt) ** gamma) * logpt
        CE_loss = K.mean(CE_loss)

        tp = K.sum(y_true * y_pred, axis=[0,1,2])
        fp = K.sum(y_pred         , axis=[0,1,2]) - tp
        fn = K.sum(y_true, axis=[0,1,2]) - tp

        score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        score = tf.reduce_mean(score)
        dice_loss = 1 - score
        return CE_loss + dice_loss
    return _dice_loss_with_Focal_Loss



def Focal_Loss(cls_weights, alpha=0.5, gamma=2):
    cls_weights = np.reshape(cls_weights, [1, 1, 1, -1])
    def _Focal_Loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        logpt = - y_true[...,:-1] * K.log(y_pred) * cls_weights
        logpt = - K.sum(logpt, axis = -1)

        pt = tf.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        CE_loss = -((1 - pt) ** gamma) * logpt
        CE_loss = K.mean(CE_loss)
        return CE_loss
    return _Focal_Loss



def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
            ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - warmup_total_iters)
                    / (total_iters - warmup_total_iters - no_aug_iter)
                )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func
