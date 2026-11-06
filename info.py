import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

print("=" * 50)
print("TensorFlow Version:", tf.__version__)

# 1. Check GPU Availability & Details
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available:", len(gpus))

if gpus:
  for i, gpu in enumerate(gpus):
    details = tf.config.experimental.get_device_details(gpu)
    print(f"  [GPU {i}] Name: {details.get('device_name', 'Unknown')}")
    print(f"  [GPU {i}] Compute Capability: {details.get('compute_capability', 'N/A')}")
    memory_info = tf.config.experimental.get_memory_info(f"GPU:{i}")
    print(f"  [GPU {i}] Current TF Memory:" f" {memory_info['current'] / (1024 ** 2):.2f} MB")
    print(f"  [GPU {i}] Peak TF Memory:" f" {memory_info['peak'] / (1024 ** 2):.2f} MB")
else:
  print("No GPU found!")
print("=" * 50)

# Optional: Enable memory growth to prevent TF from grabbing all VRAM at once
for gpu in gpus:
  try:
    tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

# 2. Turn on device placement logging to confirm GPU execution
tf.debugging.set_log_device_placement(True)

# 3. Define a workload (Large Matrix Multiplication)
# Using a size large enough to give the GPU some work to do
matrix_size = 5000
print(
    f"\nGenerating two random {matrix_size}x{matrix_size} matrices for"
    " multiplication..."
)

# Place tensors and run computation
# (By default, TensorFlow automatically assigns heavy operations to the GPU if available)
start_time = time.time()

with tf.device('/GPU:0' if gpus else '/CPU:0'):
  a = tf.random.normal([matrix_size, matrix_size], dtype=tf.float32)
  b = tf.random.normal([matrix_size, matrix_size], dtype=tf.float32)
  c = tf.matmul(a, b)
  # Force evaluation of the tensor to ensure computation finishes
  c_sum = tf.reduce_sum(c)

end_time = time.time()

print("=" * 50)
print("COMPUTATION STATS:")
print(f"Target Device: {'/GPU:0' if gpus else '/CPU:0'}")
print(f"Matrix Dimension: {matrix_size} x {matrix_size}")
print(f"Result Check (Sum): {c_sum.numpy():.4f}")
print(f"Execution Time: {end_time - start_time:.4f} seconds")
print("=" * 50)