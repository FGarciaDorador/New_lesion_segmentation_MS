import os
from operator import itemgetter

import keras.backend as K
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import tensorflow as tf


def sort_by_atlas_number(list_imgs, sort=True):
    """ Return a list of images sorted by atlas number.
    # Parameters
        list_imgs: The list that includes the names of the atlases
        sort: default True, whether the list is sorted or not (default to True)
    # Returns
        A sorted list of atlases if sort is True, otherwise, a list of tuples with the
        atlas number is returned.
    """
    numbers = []
    for i in range(len(list_imgs)):
        number = list_imgs[i][5:7]
        if number[1] == '_':
            number = number[0]
        numbers.append(int(number))

    list_imgs_tup = list(zip(list_imgs, numbers))

    if sort:
        list_imgs_tup.sort(key=itemgetter(1))
        sorted_list = [i[0] for i in list_imgs_tup]
        return sorted_list
    else:
        return list_imgs_tup


def show_slices(slices):
    """ Function to display row of image slices
    # Arguments
        slices: list of slices of 3D images to be represented
    """
    fig, axes = plt.subplots(1, len(slices), figsize=(13, 8))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


def get_lesion_activity(list_mask_bl, list_mask_fu):
    """ Return a list of lesion activities given a list of baseline and follow-up masks.
    # Arguments
        list_mask_bl: list containing the baseline masks
        list_mask_fu: list containing the follow-up masks
    # Returns
        A list of lesion activities.
    """
    lesion_activity = []
    for i in range(len(list_mask_bl)):
        mask_bl_data = list_mask_bl[i].get_fdata()
        mask_fu_data = list_mask_fu[i].get_fdata()
        la = (mask_fu_data - mask_bl_data) > 0
        lesion_activity.append(la)

    return lesion_activity


def load_data(path, img_type):
    """ Process of loading data from data path.
    # Arguments
        path: str, directory where the images are located
        img_type: 'T1' or 'FLAIR', type of image
    # Returns
        3 lists containing the baseline img_type, follow-up img_type
         and lesion activity respectively.
    """
    files = os.listdir(path)

    imgs_t1_bl = [img for img in files if '_00_T1' in img]
    imgs_t1_fu = [img for img in files if '_01_T1' in img]
    imgs_flair_bl = [img for img in files if '_00_FL' in img]
    imgs_flair_fu = [img for img in files if '_01_FL' in img]
    imgs_mask_bl = [img for img in files if '_00_lesion' in img]
    imgs_mask_fu = [img for img in files if '_01_lesion' in img]

    imgs = [imgs_t1_bl,
            imgs_t1_fu,
            imgs_flair_bl,
            imgs_flair_fu,
            imgs_mask_bl,
            imgs_mask_fu]

    imgs = [sort_by_atlas_number(i) for i in imgs]

    t1_bl = []
    t1_fu = []
    flair_bl = []
    flair_fu = []
    mask_bl = []
    mask_fu = []

    imgs_out = [t1_bl, t1_fu, flair_bl, flair_fu, mask_bl, mask_fu]
    j = 0
    for cat in imgs:
        for i in range(len(cat)):
            imgs_out[j].append(nib.load(os.path.join(path, cat[i])))
        j += 1

    lesion_activity = get_lesion_activity(mask_bl, mask_fu)

    if img_type == 'T1':
        return t1_bl, t1_fu, lesion_activity
    if img_type == 'FLAIR':
        return flair_bl, flair_fu, lesion_activity





@tf.function
def crop(atlas_number, seed_ini, x_size=128, y_size=128, z_size=128):
    """ Random crop to size (128, 128, 128).
    # Arguments
        i: Atlas number
        seed_ini : int, random initializer
        x_size: default 128, size of the cropped x axis
        y_size: default 128, size of the cropped y axis
        z_size: default 128, size of the cropped z axis
    # Returns
        5 cropped images for each atlas number: t1 baseline, t1 follow-up, flair baseline, flair follow-up
        and lesion activity respectively.
    """
    s_1 = tf.image.random_crop(t1_bl_train[atlas_number],
                               seed=seed_ini,
                               size=[x_size, y_size, z_size])
    s_2 = tf.image.random_crop(t1_fu_train[atlas_number],
                               seed=seed_ini,
                               size=[x_size, y_size, z_size])
    s_3 = tf.image.random_crop(flair_bl_train[atlas_number],
                               seed=seed_ini,
                               size=[x_size, y_size, z_size])
    s_4 = tf.image.random_crop(flair_fu_train[atlas_number],
                               seed=seed_ini,
                               size=[x_size, y_size, z_size])
    s_5 = tf.image.random_crop(lesion_activity_train[atlas_number],
                               seed=seed_ini,
                               size=[x_size, y_size, z_size])
    return s_1, s_2, s_3, s_4, s_5


def crop_preprocess(array, is_nib=True, x_crop=27, y_crop=45, z_crop=27):
    """ Preprocess nibabel images and crops symmetrically along the x, y and z axis.
    # Arguments
        array: array to be cropped
        is_nib: default True, whether the array is a nibabel object or not
        x_crop: default 27, size of the crop at the beginning and the end of the x axis of the array
        y_crop: default 45, size of the crop at the beginning and the end of the y axis of the array
        z_crop: default 27, size of the crop at the beginning and the end of the z axis of the array
    # Returns
        A new preprocessed numpy array cropped symmetrically and expanded to indicate the grayscale channel.
    """
    if is_nib:
        cropped_array = []
        for i in range(len(array)):
            img_data = array[i].get_fdata()
            if img_data.max() > 256:
                print('Intensity over 256, something wrong in position {}'.format(i))
            img_data = img_data.astype('float32')
            mean = np.mean(img_data)
            std = np.std(img_data)
            img_data /= img_data.max()  # intensity normalization
            img_data -= mean  # data centering
            img_data /= std  # data normalization

            cropped_array.append(np.array(img_data[x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]))
        return np.expand_dims(cropped_array, axis=4)
    else:
        cropped_array = np.array([array[i][x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
                                  for i in range(len(array))], dtype='float32')
        return np.expand_dims(cropped_array, axis=4)


# The original size of the images is (182, 218, 182).
