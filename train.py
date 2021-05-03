import os
import keras
import pickle

import keras.backend as K
from preprocess import load_data, crop_preprocess
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


def train(x_bl, x_fu, y_train, x_bl_val, x_fu_val, y_val, n_epochs, fusion_type, img_type, loss='dice'):
    """ Trains the model and saves the best weights and the history dictionary.
    # Arguments
        x_bl: baseline images for training
        x_fu: follow-up images for training
        y_train: lesion activity map of training images
        x_bl_val: baseline images for validation
        x_fu_val: follow-up images for validation
        y_val: lesion activity map of validation images
        n_epochs: int, number of epochs for training
        fusion_type: 'diff', 'add' or 'stack', type of fusion in the model
        img_type: 'T1' or 'FLAIR', type of image used in training
        loss: 'dice' or 'bce', loss function to be optimized
    # Returns
        Saves the best weights in .h5 file and the history dictionary.
    """
    name = img_type + '_' + fusion_type + '_' + loss
    weight_name = 'weights_' + name + '.h5'
    history_name = 'history' + name
    my_model = gessert_net(fusion_type=fusion_type)
    my_model.load_weights('weights_T1_stack.h5')
    if loss == 'bce':
        loss_function = 'binary_crossentropy'
        metric = 'accuracy'
    if loss == 'dice':
        loss_function = dice_loss
        metric = dice_coef
    my_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=loss_function,
                     metrics=[metric])
    model_checkpoint = keras.callbacks.ModelCheckpoint(weight_name,
                                                       monitor='val_loss',
                                                       save_best_only=True)
    history = my_model.fit(x=[x_bl, x_fu], y=y_train, batch_size=1, epochs=n_epochs,
                           verbose=1, validation_data=([x_bl_val, x_fu_val], y_val),
                           callbacks=[model_checkpoint])
    # Save the history dictionary
    with open(history_name, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


# The atlas number 7, 66 and 84 do not have follow-up,
# they have been manually deleted from the train and test directories.
# Specify path
data_path = 'ESTUDIO_UOC/'
train_data_path = os.path.join(data_path, 'Train')
test_data_path = os.path.join(data_path, 'Test')

# Load train and test data
print('-' * 30)
print('Loading data starts')
print('-' * 30)
t1_bl_train, t1_fu_train, lesion_activity_train = load_data(train_data_path, img_type='T1')
t1_bl_test, t1_fu_test, lesion_activity_test = load_data(test_data_path, img_type='T1')
print('-' * 30)
print('Starting the symmetric crop')
print('-' * 30)

# Crop train and test data from (182, 218, 182) to (128, 128, 128, 1) and normalize between [0, 1]
t1_bl_train = crop_preprocess(t1_bl_train)
t1_fu_train = crop_preprocess(t1_fu_train)
lesion_activity_train = crop_preprocess(lesion_activity_train, is_nib=False)

t1_bl_test = crop_preprocess(t1_bl_test)
t1_fu_test = crop_preprocess(t1_fu_test)
lesion_activity_test = crop_preprocess(lesion_activity_test, is_nib=False)

print('-' * 30)
print('Finished loading')
print('-' * 30)

train(x_bl=t1_bl_train, x_fu=t1_fu_train, y_train=lesion_activity_train,
      x_bl_val=t1_bl_test, x_fu_val=t1_fu_test, y_val=lesion_activity_test,
      n_epochs=300, fusion_type='stack', img_type='T1', loss='bce')

'''
model = gessert_net(fusion_type='stack')
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              loss=dice_loss, metrics=[dice_coef])
model.load_weights('weights_T1_stack.h5')
#t1_bl_0 = tf.convert_to_tensor(t1_bl_test[0])
#t1_fu_0 = tf.convert_to_tensor(t1_fu_test[0])
im_pred = model.predict(x=[t1_bl_test[:2], t1_fu_test[:2]])
show_slices([im_pred[0][:, :, 64], lesion_activity_test[0][:, :, 64]])
plt.savefig('result_0.png')
'''
