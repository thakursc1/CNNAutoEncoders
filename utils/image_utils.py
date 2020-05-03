from __future__ import absolute_import, division, print_function, unicode_literals

import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import models
import numpy as np
import os

# Tensor flow parameter for optimum performance for data loading
AUTOTUNE = tf.data.experimental.AUTOTUNE
print(tf.__version__)

OVER_LAY_COLOR = "grey"
OVER_LAY_COLOR_DICT = {
    "white": (228, 228, 228),
    "grey": (128, 128, 128),
}
IMAGE_SAVE_FMT = "jpeg"

IMG_WIDTH = 512
IMG_HEIGHT = 512
BATCH_SIZE = 1
TRAIN_SIZE = 0.8

DATA_DIR_PATH = "data/minst"
IMAGE_DIR_NAME = "reconstructed"


class DataIngestor:
    def __init__(self, data_dir, image_dir_name):
        self.data_dir = data_dir
        self.page_image_dir = os.path.join(self.data_dir, image_dir_name)
        self.images = [os.path.join(self.page_image_dir, i) for i in os.listdir(self.page_image_dir)]
        self.train_files = int(len(self.images) * TRAIN_SIZE)

        # Convert to tf_data_lists
        self.tf_train_image_list = tf.data.Dataset.from_tensor_slices(self.images[:self.train_files])
        self.tf_test_image_list = tf.data.Dataset.from_tensor_slices(self.images[self.train_files:])

    def get_test_images(self, size):
        tf_records = self.tf_test_image_list.map(self.TFImageIngestor)
        sample_ds = self.prepare_for_training(tf_records, batch_size=1)
        _, img = next(iter(sample_ds))
        return img

    @staticmethod
    def TFImageIngestorWithPath(file_path):
        """
        A pure TF function for scalable image transformations in the data-loader pipeline
        """
        # Read Image from path
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=1)
        img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT], method="bicubic")
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.per_image_standardization(img)
        return file_path, img

    def inference_on_sample(self, model_path):
        # Generate sample ds
        rand_imag_idx = random.randint(0, 2000)
        images = self.images[rand_imag_idx:rand_imag_idx + 10]
        tf_sample_image_list = tf.data.Dataset.from_tensor_slices(images)
        tf_records = tf_sample_image_list.map(self.TFImageIngestor)
        sample_ds = self.prepare_for_training(tf_records, batch_size=1)
        model = models.load_model(model_path)
        print("loaded model...")
        res = []
        inp = []
        # while True:
        for inp_img, out_img in iter(sample_ds):
            res.append(model.predict(inp_img))
            inp.append(inp_img.numpy())

        print("Completed Prediction...")
        show_batch(np.array(res))
        show_batch(np.array(inp))

    @staticmethod
    def TFImageIngestor(file_path):
        # Read Image from path
        img = tf.io.read_file(file_path)
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.per_image_standardization(img)
        return img, img

    @staticmethod
    def prepare_for_training(ds, batch_size=BATCH_SIZE, shuffle_buffer_size=100):
        # This is a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets that don't
        # fit in memory.
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        ds = ds.batch(batch_size)
        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def generate_train_and_test_datasets(self):
        # Generate train ds
        tf_records = self.tf_train_image_list.map(self.TFImageIngestor, num_parallel_calls=AUTOTUNE)
        train_ds = self.prepare_for_training(tf_records)

        # Generate Test ds
        tf_records = self.tf_test_image_list.map(self.TFImageIngestor, num_parallel_calls=AUTOTUNE)
        test_ds = self.prepare_for_training(tf_records)

        return train_ds, test_ds


def show_batch(image_batch):
    plt.figure(figsize=(20, 20))
    for n in range(image_batch.shape[0]):
        image = image_batch[n]
        plt.imshow(image, cmap="rgb")
        plt.axis('off')
    plt.show()


def visualize_DataIngestorInput():
    ingestor = DataIngestor(DATA_DIR_PATH, IMAGE_DIR_NAME)
    train_ds, test_ds = ingestor.generate_train_and_test_datasets()
    _, img = next(iter(train_ds))
    show_batch(img.numpy())
    return


def infer(model_path):
    ingestor = DataIngestor(DATA_DIR_PATH, IMAGE_DIR_NAME)
    ingestor.inference_on_sample(model_path)
    return


if __name__ == "__main__":
    visualize_DataIngestorInput()