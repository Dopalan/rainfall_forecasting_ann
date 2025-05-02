import tensorflow as tf
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.preprocessing import load_and_preprocess_data
import matplotlib.pyplot as plt

def repare_dataset(data_path: str):
    # Load and preprocess data
    x_train, x_test, y_train, y_test = load_and_preprocess_data(data_path)
    input_shape = x_train.shape
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    
    # Batch and shuffle dataset
    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = 100
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    
    # Buffer dataset
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    
    return train_dataset, test_dataset, input_shape
    
def build_model(input_shape):
    # Build a Sequential model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    # Model summary
    model.summary()
    
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.RMSprop(), 
              loss=tf.keras.losses.BinaryCrossentropy(), 
              metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)])
    
    return model

def train_model(model, train_dataset, test_dataset, epochs=10):
    # Train the model
    # Check if GPU is available
    if tf.config.list_physical_devices('GPU'):
        with tf.device('/GPU:0'):
            # Use GPU if available
            print("Training on GPU")
            history = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)
    else:
        # Use CPU if GPU is not available
        print("Training on CPU")
        history = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)
        
    # Save the model
    model.save('rainfall_forecasting_ann_model.keras')
    
    return history