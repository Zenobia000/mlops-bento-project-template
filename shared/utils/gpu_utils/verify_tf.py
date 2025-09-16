import tensorflow as tf
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
        print("TensorFlow is using GPU.")
    else:
        print("TensorFlow is not using GPU.")

except ImportError:
    print("TensorFlow is not installed.")