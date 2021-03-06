import pickle
import os
import keras.backend as K
import tensorflow as tf

from preprocess import get_filenames, random_crop_flip, central_crop, _set_shapes, load, load_mask, split_dataset
from model import gessert_net


def dice_coef(y_true, y_pred):
    """ Definition of the Dice coefficient.
    # Arguments
        y_true: value of the true dependent variable
        y_pred: value of the prediction of the independent variable
    # Returns
        The dice coefficient.
    """
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f, axis=-1)
    sums = K.sum(y_true_f, axis=-1) + K.sum(y_pred_f, axis=-1)
    return (2. * intersection + K.epsilon()) / (sums + K.epsilon())


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def train(dataset_train, dataset_test, n_epochs, fusion_type, img_type):
    """ Trains the model and saves the best weights and the history dictionary.
    # Arguments
        dataset_train: tf.data object containing the training samples
        dataset_test: tf.data object containing the test samples
        n_epochs: int, number of epochs for training
        fusion_type: 'diff', 'add' or 'stack', type of fusion in the model
        img_type: 'T1' or 'FLAIR', type of image used in training
        tpu: bool, default True, indicates if a tpu is used for training
    # Returns
        Saves the best weights in .h5 file and the history dictionary.
    """
    name = img_type + '_' + fusion_type
    weight_name = 'weights_' + name + '.h5'
    history_name = 'history_' + name

    my_model = gessert_net(fusion_type=fusion_type)
    my_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=dice_loss, metrics=[dice_coef])
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(weight_name, monitor='val_loss', save_best_only=True)

    history = my_model.fit(dataset_train, epochs=n_epochs, verbose=1, validation_data=dataset_test,
                           callbacks=[model_checkpoint])

    # Save the history dictionary
    with open(history_name, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


if __name__ == "__main__":
    data_path = 'ESTUDIO_UOC/'
    train_data_path = os.path.join(data_path, 'Train')
    test_data_path = os.path.join(data_path, 'Test')

    # Train dataset
    x, y = get_filenames(train_data_path, img_type='FLAIR')
    z = get_filenames(train_data_path, img_type='mask')

    path_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    mask_dataset = tf.data.Dataset.from_tensor_slices(z)
    dataset = tf.data.Dataset.zip((path_dataset, mask_dataset)).shuffle(50).repeat(10)

    ds = dataset. \
        map(lambda xx, zz: ((tf.py_function(load, [xx], [tf.float32, tf.float32])),
                            tf.py_function(load_mask, [zz], [tf.float32])),
            num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(lambda xx, zz: (tf.py_function(random_crop_flip, [xx, zz],
                                               [tf.float32, tf.float32, tf.float32])),
                num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(_set_shapes)
    ds = ds.batch(2)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    ds_train, ds_val = split_dataset(ds, 0.2)
    # train(ds_train, ds_val, n_epochs=50, fusion_type='stack', img_type='FLAIR', tpu=False)

    # EVALUATION
    # test dataset
    x_test, y_test = get_filenames(test_data_path, img_type='FLAIR')
    z_test = get_filenames(test_data_path, img_type='mask')

    path_dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    mask_dataset_test = tf.data.Dataset.from_tensor_slices(z_test)
    test_dataset = tf.data.Dataset.zip((path_dataset_test, mask_dataset_test)).shuffle(20).repeat(10)

    ds_test = test_dataset. \
        map(lambda xx, zz: ((tf.py_function(load, [xx], [tf.float32, tf.float32])),
                            tf.py_function(load_mask, [zz], [tf.float32])),
            num_parallel_calls=tf.data.AUTOTUNE)

    ds_test = ds_test.map(lambda xx, zz: (tf.py_function(central_crop, [xx, zz],
                                                         [tf.float32, tf.float32, tf.float32])),
                          num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.map(_set_shapes)
    ds_test = ds_test.batch(2)
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    model = gessert_net(fusion_type='stack')
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(1e-4, decay_steps=100000, decay_rate=1e-6)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss=dice_loss, metrics=[dice_coef])
    model.load_weights('weights_FLAIR_stack.h5')
    model.evaluate(ds_test)
