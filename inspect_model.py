import tensorflow as tf
import os

MODEL_PATH = os.path.join("models", "dogcatmodel.h5")

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at: {MODEL_PATH}")
else:
    print(f"Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Print the model summary
    print("\n--- MODEL SUMMARY ---")
    model.summary()
    print("\n--- END SUMMARY ---\n")
    
    # Extract the input shape
    try:
        input_shape = model.input_shape
        print(f"The model expects an input shape of: {input_shape}")
        print(f"This means your IMAGE_SIZE in main.py should be: ({input_shape[1]}, {input_shape[2]})")
    except Exception as e:
        print(f"Could not automatically determine input shape: {e}")