import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
from matplotlib.ticker import MaxNLocator
from model import gessert_net
from preprocess import show_slices, get_filenames
import nibabel as nib
import tensorflow as tf
from train import dice_coef, dice_loss


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
        mean = np.mean(img_data)
        std = np.std(img_data)
        img_data /= img_data.max()  # intensity normalization
        img_data -= mean  # data centering
        img_data /= std  # data normalization

        return np.expand_dims(img_data, axis=3)
    else:
        img_data = np.array(nib_img.get_fdata(), dtype='float32')

        return np.expand_dims(img_data, axis=3)


def load_mask(filename):
    """ Process of loading data from data path.
    # Arguments
        filename: filename corresponding to the image
    # Returns
        loaded and preprocessed array of images.
    """
    nib_image = nib.load(filename)
    mask_affine = nib_image.affine

    return preprocess_nib(nib_image, is_mask=True), mask_affine


def load(filenames):
    """ Process of loading data from data path.
    # Arguments
        filename: filename corresponding to the image
    # Returns
        loaded and preprocessed array of images.
    """
    bl, fu = filenames
    nib_image_bl = nib.load(bl)
    nib_image_fu = nib.load(fu)
    bl_preprocessed = preprocess_nib(nib_image_bl, is_mask=False)
    fu_preprocessed = preprocess_nib(nib_image_fu, is_mask=False)

    return bl_preprocessed, fu_preprocessed


def central_crop(image, x_crop=27, y_crop=45, z_crop=27):
    """ crops symmetrically along the x, y and z axis.
    # Arguments
        image: image to be cropped.

        x_crop: int, default 27, size to crop at the beginning and end of the x axis.
        y_crop: int, default 45, size to crop at the beginning and end of the y axis.
        z_crop: int, default 27, size to crop at the beginning and end of the z axis.
    # Returns
        3 arrays (baseline image, follow-up image and segmentation mask) cropped symmetrically.
    """
    image = image[x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop, :]

    return image


def plot_loss_metrics(history_file):
    """ Plots the loss function and metrics of a trained model.
    # Arguments
        history_file: value of the true dependent variable
    # Returns
        A plot with the loss and metric curves.
    """
    history = pickle.load(open(history_file, "rb"))
    loss, metric, val_loss, val_metric = islice(history.keys(), 4)
    n_epochs = len(history[loss])

    plt.style.use("ggplot")
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(13, 8))

    ax1.set_title(loss)
    ax1.plot(np.arange(1, n_epochs + 1), history[loss], label='train')
    ax1.plot(np.arange(1, n_epochs + 1), history[val_loss], label='test')
    ax1.legend()

    ax2.set_title(metric)
    ax2.plot(np.arange(1, n_epochs + 1), history[metric], label='train')
    ax2.plot(np.arange(1, n_epochs + 1), history[val_metric], label='test')
    ax2.set_xlabel('Epochs')
    ax2.set_xlim((1, n_epochs + 1))
    xa = ax2.get_xaxis()
    xa.set_major_locator(MaxNLocator(integer=True))
    ax2.legend()
    plt.savefig(history_file + '.png')
    plt.show()


def predict_and_save(number, bl, fu, la, test=False):
    if test:
        if number >= 5:
            j = number + 62
        else:
            j = number + 61
    else:
        if number >= 6:
            j = number + 2
        else:
            j = number + 1

    baseline, follow_up = load((bl[number], fu[number]))
    mask, affine = load_mask(la[number])

    baseline = central_crop(baseline)
    follow_up = central_crop(follow_up)
    mask = central_crop(mask)

    my_model = gessert_net(fusion_type='stack')
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(1e-4, decay_steps=100000, decay_rate=1e-6)
    my_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                     loss=dice_loss, metrics=[dice_coef])
    my_model.load_weights('weights_FLAIR_stack.h5')

    im_pred = my_model.predict(x=[baseline.reshape((1, 128, 128, 128, 1)),
                                  follow_up.reshape((1, 128, 128, 128, 1))])
    im_pred = im_pred[0, :, :, :, 0]
    result = np.zeros((182, 218, 182))
    result[27:-27, 45:-45, 27:-27] = im_pred
    nft_img = nib.Nifti1Image(np.array(result, dtype='float32'), affine)
    name = 'atlas' + str(j) + '_lesion_activity_predicted.nii.gz'
    nib.save(nft_img, name)
    show_slices([im_pred[:, :, 64], mask[:, :, 64, 0],
                 baseline[:, :, 64, 0], follow_up[:, :, 64, 0]])
    plt.savefig('result_atlas_' + str(j) + '.png')


if __name__ == '__main__':
    # plot_loss_metrics('history_FLAIR_stack_90')

    data_path = 'ESTUDIO_UOC/'
    train_data_path = os.path.join(data_path, 'Train')
    test_data_path = os.path.join(data_path, 'Test')

    # Train dataset
    x, y = get_filenames(test_data_path, img_type='FLAIR')
    z = get_filenames(test_data_path, img_type='mask')

    for i in range(len(x)):
        predict_and_save(i, x, y, z, test=True)
