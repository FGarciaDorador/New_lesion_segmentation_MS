# Multiple Sclerosis Lesion Activity Segmentation with Two-Path CNNs
In this repository we have developed a model to detect new MS lesions based on the paper: *Multiple Sclerosis Lesion Activity Segmentation with Attention-Guided Two-Path CNNs* (https://www.sciencedirect.com/science/article/abs/pii/S0895611120300732). The model has been developed using python 3.8 and tensorflow 2.5.

The architecture of the model is well summarized in the following image: ![image](2_paths_CNN.png) 

The model takes the baseline and follow-up scans of size 128 x 128 x 128 and outputs the segmentation of the lesion activity between the 2 time-points.

## Structure

### preprocess.py
This script contains all the functions that were necesary to create our tf.data.Dataset object, as well as the functions to augment our data. The functions used are:
+  **get_filenames**: gets the filenames of the type of image you want (FLAIR, T1, or lesion mask at baseline or follow-up time-points).
+  **sort_by_atlas_number**: sort the arrays with filenames by atlas number.
+  **preprocess_nib**: gets the data of a nibabel object and preprocesses it (normalization of intensity and standarisation) and expands 1 dimension to indicate that it is a grayscale image.
+  **load and load_mask**: these functions load the .nii.gz images and return them as preproccessed 4 dimensional arrays.
+  **img_shift and get_lesion_activity**: these functions are used to create the lesion activity mask from the baseline and follow-up segmentation masks. They remove those voxels with less than 2 voxels of volume. This action is performed because the baseline and follow-up scans do not match pixel by pixel.
+  **random_crop_flip**: used for data augmentation. Random crops to subvolumes of 128 x 128 x 128 and flips randomly across the x, y or z axis.
+  **central_crop**: used for prediction and evaluation. Crops symmetrically to 128 x 128 x 128 along the x, y and z axis.
+  **_set_shapes**: use to indicate the shape of the tensors inside the tf.data.Dataset object.
+  **split_dataset**: used before training. Splits the tf.data.Dataset object in training and validation datasets.

### model.py
This script contains the Two-Path CNNs. The functions inside the script are:
+  **identity_block**: part of the Resblock that does not have a convolutional layer at shortcut.
+  **conv_block**: part of the Resblock that has a convolutional layer at shortcut.
+  **encoder**: part of the net that encodes the images from (128, 128, 128, 1) to (16, 16, 16, 128).
+  **fusion**: determines the fusion strategy of the two encoded images (addition, difference and stack).
+  **decoder**: part of the net that takes the fused images and creates the mask by convoluting upwards to (128, 128, 128, 1)
+  **gessert_net**: combines the previous elements to create the architechture.

### train.py
This script contains the creation of the tf.data.Dataset object for training and evaluation. It also includes the functions to create the metric, loss, and train process:
+  **dice_coef**: used as a metric. Coefficient that indicates how much overlapped are the predicted mask and the actual mask.
+  **dice_loss**: used as the loss function to optimize. Defined as 1 - dice_coeff.
+  **train**: this function simplifies the training process. It compiles the model, saving the best weights and the history dictionary.

### results.py
This script contains the functions to evaluate and predict, as well as some functions to preprocess the data adapted to arrays instead of tf.data.Dataset objects:
+  **plot_loss_metrics**: plots the evolution of training per epoch.
+  **predict_and_save**: predicts the lesion activity mask, saves it as .nii.gz and creates an image with a central slice comparing it with the gold standard.
