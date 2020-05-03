import os
import shutil
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Input, \
    UpSampling2D
from tensorflow.keras import Model
import tensorflow as tf
from utils.image_utils import IMG_WIDTH, IMG_HEIGHT, DataIngestor

KERNEL_SIZE = (3, 3)


class LayerVisualizerCallback(tf.keras.callbacks.Callback):
    """
    A Call back for Visualization of a particular CNN layer in TensorBoard
    """

    def __init__(self, models, layer_names, ingestorObj, logdir):
        """ Save params in constructor
        """
        super().__init__()
        self.models = models
        self.layer_names = layer_names
        self.test_image = ingestorObj.get_test_images(size=1)
        # self.file_writer_cm = tf.summary.create_file_writer(logdir + "\\cm")
        plt.figure(figsize=(20, 20))
        plt.imshow(self.test_image.numpy().reshape(IMG_WIDTH, IMG_HEIGHT), cmap="gray")
        plt.axis("off")
        plt.show()

    def on_epoch_end(self, epoch, logs={}):
        if epoch % 5 == 0:
            for layer in self.layer_names:
                output = self.model.get_layer(layer).output
                intermediate_layer_model = Model(inputs=self.model.input,
                                                 outputs=output)
                output = intermediate_layer_model.predict(self.test_image)
                if output.shape[3] > 1:
                    output = output.mean(axis=3)
                output = output.reshape(output.shape[1], output.shape[2])
                plt.figure(figsize=(20, 20))
                plt.imshow(output, cmap="gray")
                plt.axis("off")
                plt.show()
                # tf.summary.image("{}_viz".format(layer), tf.convert_to_tensor(output), step=epoch)


def AutoEncoder():
    inp_x = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1), name="Input")

    # Encoder model
    x = Conv2D(128, KERNEL_SIZE, padding="same", activation="relu", name="Conv1")(inp_x)
    x = MaxPooling2D(pool_size=(2, 2), name="MaxPool1")(x)
    x = Conv2D(64, KERNEL_SIZE, padding="same", activation="relu", name="Conv2")(x)
    x = MaxPooling2D(pool_size=(2, 2), name="MaxPool2")(x)
    x = Conv2D(32, KERNEL_SIZE, padding="same", activation="relu", name="Conv3")(x)
    x = MaxPooling2D(pool_size=(2, 2), name="MaxPool3")(x)
    x = Conv2D(16, KERNEL_SIZE, padding="same", activation="relu", name="Conv4")(x)
    x = MaxPooling2D(pool_size=(2, 2), name="MaxPool4")(x)
    x = Conv2D(8, KERNEL_SIZE, padding="same", activation="relu", name="Conv5")(x)

    # Latent Space
    x = MaxPooling2D(pool_size=(2, 2), name="MaxPool5")(x)

    # Decoder model
    x = Conv2DTranspose(8, KERNEL_SIZE, padding="same", activation="relu", name="DeConv5")(x)
    x = UpSampling2D(size=(2, 2), name="Upsample5")(x)
    x = Conv2DTranspose(16, KERNEL_SIZE, padding="same", activation="relu", name="DeConv4")(x)
    x = UpSampling2D(size=(2, 2), name="Upsample4")(x)
    x = Conv2DTranspose(32, KERNEL_SIZE, padding="same", activation="relu", name="DeConv3")(x)
    x = UpSampling2D(size=(2, 2), name="Upsample3")(x)
    x = Conv2DTranspose(64, KERNEL_SIZE, padding="same", activation="relu", name="DeConv2")(x)
    x = UpSampling2D(size=(2, 2), name="UpSample2")(x)
    x = Conv2DTranspose(128, KERNEL_SIZE, padding="same", activation="relu", name="DEConv1")(x)
    x = UpSampling2D(size=(2, 2), name="Upsample1")(x)

    # 1696X1100X1
    output = Conv2D(1, (1, 1), padding="same", name="DeConv_Last")(x)
    model = Model(inputs=inp_x, outputs=output)

    model.compile(optimizer='adam',
                  loss='mse')
    print(model.summary())
    return model


def image_resize(x):
    return tf.image.resize(x, [IMG_WIDTH, IMG_HEIGHT])


if __name__ == "__main__":

    # Using Image Utils process and make a TFDataSet Iterator
    ingestor = DataIngestor("data/img", "mnist")
    train_ds, test_ds = ingestor.generate_train_and_test_datasets()

    # Initialize a new Model
    vcae = AutoEncoder()

    # Setup Log directory for Tensor board
    log_dir = "data\\models\\logs\\512_latent_5_conv"

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.mkdir(log_dir)
    model_dir = log_dir + "\\models"
    os.mkdir(model_dir)
    model_names = model_dir + "\\{}".format("model_{epoch}.hdf5")

    # add layer visualizer
    layer_viz = LayerVisualizerCallback(vcae, layer_names=["DeConv_Last"], ingestorObj=ingestor, logdir=log_dir)

    # Add callbacks for training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=0, mode='auto',
                                         baseline=None, restore_best_weights=False),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=False),
        tf.keras.callbacks.ModelCheckpoint(model_names, monitor='val_loss', verbose=0, save_best_only=True,
                                           save_weights_only=False, mode='auto', period=1),
        layer_viz
    ]

    vcae.fit(train_ds, epochs=100, validation_data=test_ds, shuffle=True, callbacks=callbacks)
