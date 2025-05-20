from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import layers
import tensorflow as tf
from model.convnext import *

from utils import try_count_flops, get_model_size
import h5py
# ------------------------------------------------------------------------------
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.98
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
# ------------------------------------------------------------------------------

def MultiScaleDWConv(input, block_id):
    projection_dims = input.shape[-1]

    x_1 = layers.Conv2D(
        filters=projection_dims,
        kernel_size=3,
        padding="same",
        groups=projection_dims,
        name=f'convnext_decoder_{block_id}_scale_3' + "_depthwise_conv")(input)

    x_2 = layers.Conv2D(
        filters=projection_dims,
        kernel_size=5,
        padding="same",
        groups=projection_dims,
        name=f'convnext_decoder_{block_id}_scale_5' + "_depthwise_conv")(input)

    x_3 = layers.Conv2D(
        filters=projection_dims,
        kernel_size=7,
        padding="same",
        groups=projection_dims,
        name=f'convnext_decoder_{block_id}_scale_7' + "_depthwise_conv")(input)

    x_4 = layers.Conv2D(
        filters=projection_dims,
        kernel_size=9,
        padding="same",
        groups=projection_dims,
        name=f'convnext_decoder_{block_id}_scale_9' + "_depthwise_conv")(input)

    out = Add()([x_1, x_2, x_3, x_4])

    return out


def convnext_block(x, num_filters, block_id):
    """
        ConvNeXt Block. There are two equivalent implementations:
       (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
       (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back

       Returns:
       A function representing a ConvNeXt-SE block.
    """

    # DwConv -> Layer Norm ->  1x1 Conv -> SE -> GELU -> 1x1 Conv
    # --------------------------------------------------------------------------------
    residual = x
    x = MultiScaleDWConv(x, block_id)

    #
    x = layers.LayerNormalization(epsilon=1e-5, name=f'convnext_decoder_{block_id}' + "_layernorm")(x)
    x = layers.Dense(4 * num_filters, name=f'convnext_decoder_{block_id}' + "_pointwise_conv_1")(x)

    x = layers.Activation("gelu", name=f'convnext_decoder_{block_id}' + "_gelu")(x)
    x = layers.Dense(num_filters, name=f'convnext_decoder_{block_id}' + "_pointwise_conv_2")(x)
    # --------------------------------------------------------------------------------

    # Shortcut Path
    residual = layers.Conv2D(
               filters=num_filters,
               kernel_size=1,
               padding='same',
               kernel_initializer='he_normal',
               name=f'convnext_decoder_{block_id}' + "_identity")(residual)
    residual = layers.LayerNormalization(epsilon=1e-5, name=f'convnext_decoder_{block_id}' + "_identity_layernorm")(
               residual)

    return add([x, residual], name=f'convnext_decoder_{block_id}' + "_add")



    # --------------------------------------------------------------------------------


# <=============================== Up-Concat-ConvNext block. ================================>
def decoder_block(input, skip_features, num_filters, block_id):
    """
       Decoder block with modified ConvNeXt-SE block.
    """
    x = UpSampling2D((2, 2))(input)
    x = Concatenate()([x, skip_features])
    x = convnext_block(x, num_filters, block_id)
    #x = conv_block(x,num_filters)

    return x



def ConvNeXt_Unet(input_shape=(256, 256, 3), num_classes=4, backbone='ConvNeXtTiny', name='ConvUNeXt'):
    """ INPUT """
    inputs = Input(shape=input_shape)


    up_filter = [256, 128, 64, 16]

    if backbone == 'ConvNeXtTiny':
        encoder = ConvNeXtTiny(include_top=False, weights='imagenet', input_tensor=inputs, include_preprocessing=False)
    else:
        raise ValueError('Unsupported backbone type `{}`, Except for VGG , ResNet50 or ConvNeXt.'.format(backbone))

    assert isinstance(encoder, Model), 'The model should be the instantiation of the keras Model '

    # ------------------------------------------------------------------------------------------------------------------
    """
    | ConvNeXtTiny:           | ConvNeXtSmall:           | ConvNeXtBase:           | ConvNeXtLarge:
    | s1 -> index=25          | s1 -> index=25           | s1 -> index=25          | s1 -> index=25
    | s2 -> index=50          | s2 -> index=50           | s2 -> index=50          | s2 -> index=50
    | s3 -> index=123         | s3 -> index=267          | s3 -> index=267         | s3 -> index=267
    | b1 -> index=148         | b1 -> index=292          | b1 -> index=292         | b1 -> index=292
    """
    # ------------------------------------------------------------------------------------------------------------------
    """ Encoder """
    s1 = encoder.get_layer(index=25).output
    s2 = encoder.get_layer(index=50).output
    s3 = encoder.get_layer(index=123).output

    """ Bridge """
    b1 = encoder.get_layer(index=148).output     # feature size: 8x8

    """ Decoder """
    d1 = decoder_block(b1, s3, up_filter[0], 1)  # feature size: 16x16
    d2 = decoder_block(d1, s2, up_filter[1], 2)  # feature size: 32x32
    d3 = decoder_block(d2, s1, up_filter[2], 3)  # feature size: 64x64
    d5_up = UpSampling2D((4, 4))(d3)             # feature size: 256x256
    d5 = convnext_block(d5_up, up_filter[3], 4)



    """ Output """
    outputs = Conv2D(filters=num_classes, kernel_size=(1, 1), strides=1, padding='same', activation='softmax',
                      kernel_initializer='he_normal', name='projection_head')(d5)

    return Model(inputs, outputs, name=name)



if __name__ == "__main__":

    count_flops_model = ConvNeXt_Unet()
