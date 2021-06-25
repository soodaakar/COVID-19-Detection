import os
import pathlib
from PIL import Image

import IPython.display as display
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Softmax)
import tensorflow_hub as hub

import datetime

print("GPU Available: ", tf.config.list_physical_devices('GPU'))

IMG_HEIGHT = 600
IMG_WIDTH = 600
IMG_CHANNELS = 3

BATCH_SIZE = 8
# 10 is a magic number tuned for local training of this dataset.
SHUFFLE_BUFFER = 10 * BATCH_SIZE
AUTOTUNE = tf.data.experimental.AUTOTUNE

VALIDATION_IMAGES = 370
VALIDATION_STEPS = VALIDATION_IMAGES // BATCH_SIZE

def decode_img(img, reshape_dims):
    # Convert the compressed string to a 3D uint8 tensor.
    img = tf.image.decode_jpeg(img, channels=IMG_CHANNELS)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Resize the image to the desired size.
    return tf.image.resize(img, reshape_dims)

CLASS_NAMES = ['Typical_Appearance', 'Negative_for_Pneumonia','Indeterminate_Appearance', 'Atypical_Appearance']

def decode_csv(csv_row):
    record_defaults = ["path", "target"]
    filename, label_string = tf.io.decode_csv(csv_row, record_defaults)
    image_bytes = tf.io.read_file(filename=filename)
    label = tf.math.equal(CLASS_NAMES, label_string)
    return image_bytes, label

MAX_DELTA = 63.0 / 255.0  # Change brightness by at most 17.7%
CONTRAST_LOWER = 0.2
CONTRAST_UPPER = 1.8


def read_and_preprocess(image_bytes, label, random_augment=False):
    if random_augment:
        img = decode_img(image_bytes, [IMG_HEIGHT + 10, IMG_WIDTH + 10])
        img = tf.image.random_crop(img, [IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, MAX_DELTA)
        img = tf.image.random_contrast(img, CONTRAST_LOWER, CONTRAST_UPPER)
    else:
        img = decode_img(image_bytes, [IMG_WIDTH, IMG_HEIGHT])
    return img, label


def read_and_preprocess_with_augment(image_bytes, label):
    return read_and_preprocess(image_bytes, label, random_augment=True)

def load_dataset(csv_of_filenames, batch_size, training=True):
    dataset = tf.data.TextLineDataset(filenames=csv_of_filenames) \
        .map(decode_csv).cache()

    if training:
        dataset = dataset \
            .map(read_and_preprocess_with_augment) \
            .shuffle(SHUFFLE_BUFFER) \
            .repeat(count=None)  # Indefinately.
    else:
        dataset = dataset \
            .map(read_and_preprocess) \
            .repeat()  

    # Prefetch prepares the next set of batches while current batch is in use.
    return dataset.batch(batch_size=batch_size).prefetch(buffer_size=AUTOTUNE)

train_path = "gs://qwiklabs-gcp-03-365bf9c0599c-kaggle/train_data_image_classification.txt"
eval_path = "gs://qwiklabs-gcp-03-365bf9c0599c-kaggle/val_data_image_classification.txt"
nclasses = len(CLASS_NAMES)
hidden_layer_1_neurons = 400
hidden_layer_2_neurons = 100
dropout_rate = 0.25
num_filters_1 = 64
kernel_size_1 = 3
pooling_size_1 = 2
num_filters_2 = 32
kernel_size_2 = 3
pooling_size_2 = 2

# layers = [
#     Conv2D(num_filters_1, kernel_size=kernel_size_1,
#            activation='relu',
#            input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)),
#     MaxPooling2D(pooling_size_1),
#     Conv2D(num_filters_2, kernel_size=kernel_size_2,
#            activation='relu'),
#     MaxPooling2D(pooling_size_2),
#     Flatten(),
#     Dense(hidden_layer_1_neurons, activation='relu'),
#     Dense(hidden_layer_2_neurons, activation='relu'),
#     Dropout(dropout_rate),
#     Dense(nclasses),
#     Softmax()
# ]

# old_model = Sequential(layers)
# old_model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy',
#     metrics=['accuracy'])

train_ds = load_dataset(train_path, BATCH_SIZE)
eval_ds = load_dataset(eval_path, BATCH_SIZE, training=False)

strategy = tf.distribute.MirroredStrategy()

# module_selection = "mobilenet_v2_100_224"
module_handle = "https://tfhub.dev/tensorflow/efficientnet/b7/classification/1"

NOW = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

checkpoint_path_1 = "gs://qwiklabs-gcp-03-365bf9c0599c-kaggle/covid_classifier/{}/model.ckpt".format(NOW)
checkpoint_dir_1 = os.path.dirname(checkpoint_path_1)

# Create a callback that saves the model's weights
checkpoint_callback_1 = tf.keras.callbacks.ModelCheckpoint(
   checkpoint_path_1, verbose=1, save_weights_only=True,
   # Save weights, save_best_only=every epoch.
   save_freq='epoch')

tensorboard_path = "gs://qwiklabs-gcp-03-365bf9c0599c-kaggle/covid_classifier/{}/tensorboard".format(NOW)
tensorboard_cb = tf.keras.callbacks.TensorBoard(tensorboard_path,
                                       histogram_freq=1)

with strategy.scope():
    transfer_model = tf.keras.Sequential([
        hub.KerasLayer(module_handle, trainable=False),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(
            nclasses,
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    ])
    transfer_model.build((None,)+(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    transfer_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])
    
transfer_model.summary()

transfer_model.fit(
    train_ds,
    epochs=50,
    steps_per_epoch=100,
    validation_data=eval_ds,
    validation_steps=VALIDATION_STEPS,
    callbacks=[checkpoint_callback_1, tensorboard_cb]
)

model_path = "gs://qwiklabs-gcp-03-365bf9c0599c-kaggle/covid_classifier/{}/model".format(NOW)

tf.saved_model.save(transfer_model, model_path)
