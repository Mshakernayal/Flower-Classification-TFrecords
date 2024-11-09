# Flower-Classification-with-TPUs

This code supports running on Kaggle TPUs.

## 1.Enable TPU from the settings of the notebook

## 2.Verify TPU Setup

The following code can be used to check whether the TPU is available after it has been enabled:

```python
import tensorflow as tf
print("TPU devices:", tf.config.experimental.list_logical_devices('TPU'))
```

## 3.Use TPU Strategy

When working with a TPU, it's essential to enclose your model training code within a `tf.distribute.TPUStrategy()` context to effectively distribute computations across the TPU cores.

```python
import tensorflow as tf

# Create a TPU strategy
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

# with strategy.scope():
#     # Your model and training code here
#     model = tf.keras.Sequential([...])
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#     model.fit(training_data, training_labels, epochs=10)

```
