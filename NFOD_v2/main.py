"""
Credit to DigitalSreeni on YouTube for creating the tutorial for building this U-Net Model using tensorflow
(https://www.youtube.com/watch?v=68HR_eyzk00)
@author Caleb Gira
"""
import tensorflow as tf
import numpy as np

# Define image dimensions
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

"""
 ___________
|           |
|INPUT LAYER|
|___________|
"""
# Defines Input Layer
inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))

# Converts to integer?? (Gets rid of 'oneDNN custom operators are on...' message)
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

"""
 ________________
|                |
|CONTRACTION PATH|
|________________|
"""
# Encoder Block 1
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

# Encoder Block 2
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

# Encoder Block 3
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

# Encoder Block 4
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

# Bottleneck
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3,), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

"""
 ______________
|              |
|EXPANSIVE PATH|
|______________|
"""
# Decoder Block 1
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

# Decoder Block 2
u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

# Decoder Block 3
u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

# Decoder Block 4
u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

# Output layer: 1 filter for binary segmentation, sigmoid for probabilities
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

"""
 _____________
|             |
|LOADING MODEL|
|_____________|

"""
# Create the Keras Model
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

"""
 __________________________
|                          |
|DATA LOADING (PLACEHOLDER)|
|__________________________|

You need to load your image data (X) and corresponding masks (Y) here.
X should be a NumPy array of shape (num_samples, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
Y should be a NumPy array of shape (num_samples, IMG_WIDTH, IMG_HEIGHT, 1)

Example of how you might load and preprocess:
(This is a simplified example; actual loading will depend on your dataset structure)
"""

# --- Example Data Loading (Replace with your actual data loading logic) ---
# Assuming you have image files in 'images/' and mask files in 'masks/'
# and you've already loaded and preprocessed them into NumPy arrays.

# Placeholder: Generate some dummy data for demonstration
num_samples = 100 # Replace with the actual number of samples in your dataset
X = np.random.rand(num_samples, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS).astype(np.float32) * 255 # Simulate 0-255 images
Y = np.random.randint(0, 2, (num_samples, IMG_WIDTH, IMG_HEIGHT, 1)).astype(np.float32) # Simulate binary masks (0 or 1)

# In a real scenario, X and Y would be loaded from your dataset files (e.g., .png, .jpg)
# and then resized and normalized.
# X will be normalized by the Lambda layer already in the model.
# Y should already be 0 or 1 for binary_crossentropy.

print(f"Shape of X (Input Images): {X.shape}")
print(f"Shape of Y (Segmentation Masks): {Y.shape}")
# ------------------------------------------------------------------------

"""
 ________________
|                |
|MODEL CHECKPOINT|
|________________|

"""
# The 'MODEL_GOES_HERE' tag should be a filepath.
# It's good practice to include epoch and validation loss in the filename
# to keep track of different saved models and their performance.
# '.keras' is the recommended file format for saving entire Keras models (TensorFlow 2.x).
checkpoint_filepath = 'unet_segmentation_model_epoch{epoch:02d}_val_loss{val_loss:.4f}.keras'

checkpointer = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    verbose=1,
    save_best_only=True,
    monitor='val_loss' # Monitor validation loss to save the best model
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'), # Stop if val_loss doesn't improve for 2 epochs
    tf.keras.callbacks.TensorBoard(log_dir='logs'), # For visualizing training progress with TensorBoard
    checkpointer # Add your model checkpoint callback here
]

print("\nStarting model training...")
# 'X' is your input images, 'Y' is your corresponding segmentation masks
results = model.fit(X, Y, validation_split=0.1, batch_size=16, epochs=25, callbacks=callbacks)
print("Training completed!")

# You can access training history (loss, accuracy, etc.) from the 'results' object
print("\nTraining History:")
print(results.history.keys())
# For example, to plot loss:
# import matplotlib.pyplot as plt
# plt.plot(results.history['loss'], label='Training Loss')
# plt.plot(results.history['val_loss'], label='Validation Loss')
# plt.title('Loss over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()