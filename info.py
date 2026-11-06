import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import pynvml

# Initialize NVIDIA Management Library (NVML) to get GPU details
pynvml.nvmlInit()

print("=" * 50)
print("TensorFlow Version:", tf.__version__)

# 1. Check GPU Availability & Details
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available:", len(gpus))

if gpus:
  # Query physical specs using NVML for GPU 0
  handle = pynvml.nvmlDeviceGetHandleByIndex(0)
  card_name = pynvml.nvmlDeviceGetName(handle)
  # Handle bytes conversion safely for different pynvml versions
  if isinstance(card_name, bytes):
    card_name = card_name.decode('utf-8')

  mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle) # Note: varies by driver support, alternative is checking model specs
  temp = pynvml.nvmlDeviceGetTemperature(
    handle, pynvml.NVML_TEMPERATURE_GPU
  )

  print(f"  [GPU 0] Hardware Name: {card_name}")
  print(f"  [GPU 0] Total Physical VRAM: {mem_info.total / (1024 ** 3):.2f} GB")
  print(f"  [GPU 0] Free Physical VRAM: {mem_info.free / (1024 ** 3):.2f} GB")
  print(f"  [GPU 0] GPU Temperature: {temp}°C")
# Shut down NVML cleanly when done
pynvml.nvmlShutdown()
print("=" * 50)