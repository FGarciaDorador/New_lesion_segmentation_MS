import os
import random
from operator import itemgetter

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def img_shift(array, shift, axis):
    """ shifts an image along the indicated axis.
    # Arguments
        array: array-like image to be shifted
        shift: int, number of pixels to shift
        axis: 0, 1 or 2, the axis to shift
    # Returns
        the shifted array.
    """
    array = np.array(array, dtype=bool)
    array_neg_shift = np.roll(array, -shift, axis=axis)
    array_pos_shift = np.roll(array, shift, axis=axis)
    neg_mask = array & array_neg_shift
    pos_mask = array & array_pos_shift

    return neg_mask + pos_mask


def preprocess_nib(nib_img, is_mask=False):
    """ Preprocess nibabel images.
    # Arguments
        nib_img: nibabel image to be processed
        is_mask: default False, whether the image is a mask or not
    # Returns
        A new preprocessed numpy array expanded to indicate the grayscale channel.
    """
    if not is_mask:
        img_data = nib_img.get_fdata()
        if img_data.max() > 256:
            mask256 = img_data < 257
            img_data = img_data * mask256
        img_data = img_data.astype('float32')
        # mean = np.mean(img_data)
        # std = np.std(img_data)
        img_data /= img_data.max()  # intensity normalization
        # img_data -= mean  # data centering
        # img_data /= std  # data normalization

        return np.expand_dims(img_data, axis=3)
    else:
        img_data = np.array(nib_img.get_fdata(), dtype='float32')

        return np.expand_dims(img_data, axis=3)


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
        if list_imgs[i][12:17] == 'Train':
            ini = 23
            end = 25
        else:
            ini = 22
            end = 24

        number = list_imgs[i][ini:end]
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
        la = (list_mask_fu[i] - list_mask_bl[i]) > 0
        shift_x = img_shift(array=la, shift=1, axis=0)
        shift_y = img_shift(array=la, shift=1, axis=1)
        shift_z = img_shift(array=la, shift=1, axis=2)
        la = shift_x * shift_y * shift_z
        lesion_activity.append(la)

    return lesion_activity


def get_filenames(path, img_type):
    """ Gets the filenames of the images.
     # Arguments
        path: str, directory where the images are located
        img_type: 'T1', 'FLAIR' or 'mask', type of image
    # Returns
        2 lists containing the baseline img_type and follow-up img_type filenames.
    """
    files = os.listdir(path)
    if img_type == 'T1':
        bl_files = sort_by_atlas_number([os.path.join(path, img) for img in files if '_00_T1' in img])
        fu_files = sort_by_atlas_number([os.path.join(path, img) for img in files if '_01_T1' in img])
        return bl_files, fu_files

    if img_type == 'FLAIR':
        bl_files = sort_by_atlas_number([os.path.join(path, img) for img in files if '_00_FL' in img])
        fu_files = sort_by_atlas_number([os.path.join(path, img) for img in files if '_01_FL' in img])
        return bl_files, fu_files

    if img_type == 'mask':
        lesion_act = sort_by_atlas_number([os.path.join(path, img) for img in files if '_lesion_activity' in img])
        return lesion_act


def load(filenames):
    """ Process of loading data from data path.
    # Arguments
        filename: filename corresponding to the image
    # Returns
        loaded and preprocessed array of images.
    """
    bl, fu = filenames
    nib_image_bl = nib.load(bl.numpy().decode('utf-8'))
    nib_image_fu = nib.load(fu.numpy().decode('utf-8'))
    bl_preprocessed = preprocess_nib(nib_image_bl, is_mask=False)
    fu_preprocessed = preprocess_nib(nib_image_fu, is_mask=False)

    return bl_preprocessed, fu_preprocessed


def load_mask(filename):
    """ Process of loading data from data path.
    # Arguments
        filename: filename corresponding to the image
    # Returns
        loaded and preprocessed array of images.
    """
    nib_image = nib.load(filename.numpy().decode('utf-8'))

    return preprocess_nib(nib_image, is_mask=True)


def random_crop_flip(images, mask, width=128, height=128, depth=128):
    """ Random crops to (with, height, depth) and flips randomly along de x, y, and z axis.
    # Arguments
        images: tuple containing the baseline and follow-up images.
        mask: array, contains the segmentation mask.
        width: int, default 128, width of the cropped image.
        height: int, default 128, height of the cropped image.
        depth: int, default 128, depth of the cropped image.
    # Returns
        3 arrays (baseline image, follow-up image and segmentation mask) randomly cropped and flipped.
    """
    img_bl, img_fu = images
    img_bl = img_bl.numpy()
    img_fu = img_fu.numpy()
    mask = mask.numpy()
    x_rand = random.randint(0, img_bl.shape[1] - width)
    y_rand = random.randint(0, img_bl.shape[0] - height)
    z_rand = random.randint(0, img_bl.shape[2] - depth)
    img_bl_f = img_bl[y_rand:y_rand + height, x_rand:x_rand + width, z_rand:z_rand + depth, :]
    img_fu_f = img_fu[y_rand:y_rand + height, x_rand:x_rand + width, z_rand:z_rand + depth, :]
    mask_f = mask[0, y_rand:y_rand + height, x_rand:x_rand + width, z_rand:z_rand + depth, :]
    flip_x = random.choice([True, False])
    flip_y = random.choice([True, False])
    flip_z = random.choice([True, False])

    if flip_x:
        img_bl_f = np.flip(img_bl_f, axis=1)
        img_fu_f = np.flip(img_fu_f, axis=1)
        mask_f = np.flip(mask_f, axis=1)

    if flip_y:
        img_bl_f = np.flip(img_bl_f, axis=0)
        img_fu_f = np.flip(img_fu_f, axis=0)
        mask_f = np.flip(mask_f, axis=0)

    if flip_z:
        img_bl_f = np.flip(img_bl_f, axis=2)
        img_fu_f = np.flip(img_fu_f, axis=2)
        mask_f = np.flip(mask_f, axis=2)

    return img_bl_f, img_fu_f, mask_f


def central_crop(images, mask, x_crop=27, y_crop=45, z_crop=27):
    """ crops symmetrically along the x, y and z axis.
    # Arguments
        images: tuple containing the baseline and follow-up images.
        mask: array, contains the segmentation mask.
        x_crop: int, default 27, size to crop at the beginning and end of the x axis.
        y_crop: int, default 45, size to crop at the beginning and end of the y axis.
        z_crop: int, default 27, size to crop at the beginning and end of the z axis.
    # Returns
        3 arrays (baseline image, follow-up image and segmentation mask) cropped symmetrically.
    """
    img_bl, img_fu = images
    img_bl = img_bl.numpy()
    img_fu = img_fu.numpy()
    mask = mask.numpy()

    img_bl_f = img_bl[x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop, :]
    img_fu_f = img_fu[x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop, :]
    mask_f = mask[0, x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop, :]

    return img_bl_f, img_fu_f, mask_f


def _set_shapes(img_bl, img_fu, mask):
    """ Sets the shapes of the tensors inside tf.Dataset object.
    # Arguments
        image_bl: array, contains the baseline image.
        image_fu: array, contains the follow-up image.
        mask: array, contains the segmentation mask.
    # Returns
        A tuple containing the baseline and follow-up images, and the segmentation mask.
    """
    img_bl.set_shape([128, 128, 128, 1])
    img_fu.set_shape([128, 128, 128, 1])
    mask.set_shape([128, 128, 128, 1])

    return (img_bl, img_fu), mask


# The original size of the images is (182, 218, 182).



