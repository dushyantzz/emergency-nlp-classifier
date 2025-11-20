import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("\nGPU Devices:", tf.config.list_physical_devices('GPU'))
print("All Devices:", tf.config.list_physical_devices())

# Check CUDA availability
from tensorflow.python.client import device_lib
print("\nDevice Library:")
print(device_lib.list_local_devices())
