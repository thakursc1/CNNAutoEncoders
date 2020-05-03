# CNN based AutoEncoders 
:test_tube: An Implementation of Autoencoders with TF2.0

## Usage 



```python
>>> # Using Image Utils process and make a TFDataSet Iterator
>>> ingestor = DataIngestor("data/img", "mnist")
>>> train_ds, test_ds = ingestor.generate_train_and_test_datasets()

>>> # Initialize a new Model
>>> vcae = AutoEncoder()
>>> # Add callbacks for training
>>> callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=0, mode='auto',
                                         baseline=None, restore_best_weights=False),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=False),
        tf.keras.callbacks.ModelCheckpoint(model_names, monitor='val_loss', verbose=0, save_best_only=True,
                                           save_weights_only=False, mode='auto', period=1),
        layer_viz
    ]
>>> vcae.fit(train_ds, epochs=100, validation_data=test_ds, shuffle=True, callbacks=callbacks)
```

## Dependencies 
```
keras
matplotlib
Tensorflow 2.0 
```
