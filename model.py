import tensorflow_addons as tfa
from keras import layers
from keras import Model
import tensorflow as tf


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    in_name_base = 'in' + str(stage) + block + '_branch'

    x = layers.Conv3D(filters1, (1, 1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = tfa.layers.InstanceNormalization(axis=4,
                                         center=True,
                                         scale=True,
                                         beta_initializer="random_uniform",
                                         gamma_initializer="random_uniform",
                                         name=in_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv3D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = tfa.layers.InstanceNormalization(axis=4,
                                         center=True,
                                         scale=True,
                                         beta_initializer="random_uniform",
                                         gamma_initializer="random_uniform",
                                         name=in_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv3D(filters3, (1, 1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = tfa.layers.InstanceNormalization(axis=4,
                                         center=True,
                                         scale=True,
                                         beta_initializer="random_uniform",
                                         gamma_initializer="random_uniform",
                                         name=in_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    in_name_base = 'in' + str(stage) + block + '_branch'

    x = layers.Conv3D(filters1, (1, 1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = tfa.layers.InstanceNormalization(axis=4,
                                         center=True,
                                         scale=True,
                                         beta_initializer="random_uniform",
                                         gamma_initializer="random_uniform",
                                         name=in_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv3D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = tfa.layers.InstanceNormalization(axis=4,
                                         center=True,
                                         scale=True,
                                         beta_initializer="random_uniform",
                                         gamma_initializer="random_uniform",
                                         name=in_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv3D(filters3, (1, 1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = tfa.layers.InstanceNormalization(axis=4,
                                         center=True,
                                         scale=True,
                                         beta_initializer="random_uniform",
                                         gamma_initializer="random_uniform",
                                         name=in_name_base + '2c')(x)

    shortcut = layers.Conv3D(filters3, (1, 1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = tfa.layers.InstanceNormalization(axis=4,
                                                center=True,
                                                scale=True,
                                                beta_initializer="random_uniform",
                                                gamma_initializer="random_uniform",
                                                name=in_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def encoder(input_tensor, name):
    """Part of the net that encodes the image.
    # Arguments
        input_tensor: input_tensor
        name: string, name of the encoding block, used for generating layer names
    # Returns
        output tensor of the encoding.
    """
    x = layers.Conv3D(16, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                      padding='same', name='Conv_in' + name)(input_tensor)
    x = tfa.layers.InstanceNormalization(axis=4,
                                         center=True,
                                         scale=True,
                                         beta_initializer="random_uniform",
                                         gamma_initializer="random_uniform",
                                         name='in1a' + name)(x)
    x = layers.Activation('relu')(x)
    x = identity_block(x, 3, [4, 4, 16], stage=1, block='b' + '_' + name)

    x = conv_block(x, 3, [8, 8, 32], stage=2, block='a' + '_' + name)
    x = identity_block(x, 3, [8, 8, 32], stage=2, block='b' + '_' + name)
    x = identity_block(x, 3, [8, 8, 32], stage=2, block='c' + '_' + name)

    x = conv_block(x, 3, [16, 16, 64], stage=3, block='a' + '_' + name)
    x = identity_block(x, 3, [16, 16, 64], stage=3, block='b' + '_' + name)
    x = identity_block(x, 3, [16, 16, 64], stage=3, block='c' + '_' + name)

    x = conv_block(x, 3, [32, 32, 128], stage=4, block='a' + '_' + name)
    x = identity_block(x, 3, [32, 32, 128], stage=4, block='b' + '_' + name)
    x = identity_block(x, 3, [32, 32, 128], stage=4, block='c' + '_' + name)
    x = identity_block(x, 3, [32, 32, 128], stage=4, block='d' + '_' + name)
    x = identity_block(x, 3, [32, 32, 128], stage=4, block='e' + '_' + name)

    return x


def fusion(tensor_bl, tensor_fu, fusion_type):
    """function that combines the coded tensors of the baseline and follow-up images.
    # Arguments
        tensor_bl: tensor, the baseline image after codification
        tensor_fu: tensor, the follow-up image after codification
        fusion_type: 'diff', 'add' or 'stack', type of fusion
    # Returns
        A tensor resulted from the fusion operation.
    """
    if fusion_type == 'diff':
        x = layers.subtract([tensor_fu, tensor_bl], name='fusion')
        return x
    if fusion_type == 'add':
        x = layers.add([tensor_fu, tensor_bl], name='fusion')
        return x
    if fusion_type == 'stack':
        x = layers.Concatenate(axis=4, name='fusion')([tensor_fu, tensor_bl])
        return x


def decoder(input_tensor):
    """Part of the net that decodes the image.
    # Arguments
        input_tensor: input_tensor
    # Returns
        A tensor of the same size that the training input.
    """
    x = layers.Conv3DTranspose(128, kernel_size=(3, 3, 3), strides=(2, 2, 2), name='Conv_up',
                               padding='same')(input_tensor)
    x = tfa.layers.InstanceNormalization(axis=4,
                                         center=True,
                                         scale=True,
                                         beta_initializer="random_uniform",
                                         gamma_initializer="random_uniform",
                                         name='in5a')(x)
    x = layers.Activation('relu')(x)
    x = identity_block(x, 3, [32, 32, 128], stage=5, block='b')

    x = layers.Conv3DTranspose(64, kernel_size=(3, 3, 3), strides=(2, 2, 2), name='Conv_up_2',
                               padding='same')(x)
    x = tfa.layers.InstanceNormalization(axis=4,
                                         center=True,
                                         scale=True,
                                         beta_initializer="random_uniform",
                                         gamma_initializer="random_uniform",
                                         name='in6a')(x)
    x = layers.Activation('relu')(x)
    x = identity_block(x, 3, [16, 16, 64], stage=6, block='b')

    x = layers.Conv3DTranspose(32, kernel_size=(3, 3, 3), strides=(2, 2, 2), name='Conv_up_3',
                               padding='same')(x)
    x = tfa.layers.InstanceNormalization(axis=4,
                                         center=True,
                                         scale=True,
                                         beta_initializer="random_uniform",
                                         gamma_initializer="random_uniform",
                                         name='in7a')(x)
    x = layers.Activation('relu')(x)
    x = identity_block(x, 3, [8, 8, 32], stage=7, block='b')

    x = layers.Conv3D(1, kernel_size=(1, 1, 1), name='Conv_out')(x)
    x = layers.Activation('sigmoid')(x)

    return x


def gessert_net(fusion_type):
    """Implementation of the attention guided two-path CNN.
    # Arguments
        fusion_type: 'diff', 'add' or 'stack', type of fusion
    # Returns
        A model based on the paper by Gessert et al.
        'Multiple Sclerosis Lesion Activity Segmentation with Attention-Guided Two-Path CNNs'
        'https://www.sciencedirect.com/science/article/abs/pii/S0895611120300732'
    """
    input_bl = layers.Input((128, 128, 128, 1))
    input_fu = layers.Input((128, 128, 128, 1))

    encoded_bl = encoder(input_bl, name='bl')
    encoded_fu = encoder(input_fu, name='fu')

    union = fusion(encoded_bl, encoded_fu, fusion_type=fusion_type)

    output = decoder(union)

    return Model(inputs=[input_bl, input_fu], outputs=output)


'''
model = gessert_net(fusion_type='stack')
tf.keras.utils.plot_model(
    model, to_file='gessert_net.png', show_shapes=True, show_dtype=False,
    show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
)
'''
