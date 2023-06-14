import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

labels = os.listdir("train")
labels

from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_path = "train"
image_height, image_width = 400, 400
train_data = ImageDataGenerator(validation_split=0.2)

# Load the training dataset
train_gen = train_data.flow_from_directory(
    dataset_path,
    target_size=(image_height, image_width),
    batch_size=32,
    shuffle=True,
    class_mode='categorical',
    subset='training'
)

val_data = ImageDataGenerator(validation_split=0.2)

# Load the validation dataset
val_gen = val_data.flow_from_directory(
    dataset_path,
    target_size=(image_height, image_width),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

#Data augmentation to improve the model's ability to generalize and handle variations in the input data

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

train_dataset_augmented = datagen.flow_from_directory(
    dataset_path,
    target_size=(image_height, image_width),
    batch_size=32,
    class_mode='categorical'
)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import Sequential

# Adjust the model architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(400, 400, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))

# Output branch for drowsiness detection
drowsiness_output = Dense(4, activation='softmax', name='drowsiness_output')(model.output)

# Output branch for yawn detection
yawn_output = Dense(4, activation='softmax', name='yawn_output')(model.output)

# Create the combined model
model = Model(inputs=model.input, outputs=[drowsiness_output, yawn_output])

# Compile the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
model.summary()

# Model Training
history = model.fit(train_gen, epochs=1, validation_data=(val_gen, None))

test_data = ImageDataGenerator(rescale=1./255)
test_dataset_path="test"

test_gen = test_data.flow_from_directory(
    test_dataset_path,
    target_size=(image_height, image_width),
    batch_size=32,
    shuffle=False,
    class_mode='categorical'
)

evaluation_results = model.evaluate(test_gen)
test_loss = evaluation_results[0]
test_accuracy = evaluation_results[1]

print("Loss:", test_loss)
print("Accuracy:", test_accuracy)

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Check if the frame is valid
    if not ret:
        break

    # Resize the frame to match the desired input shape
    frame = cv2.resize(frame, (400, 400))

    # Preprocess the frame (normalize, etc.)
    # ...

    # Make predictions
    predictions = model.predict(np.expand_dims(frame, axis=0))
    drowsiness_prediction = predictions[0][0]
    yawn_prediction = predictions[1][0]

    # Convert the frame to BGR color format
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Determine the labels based on the threshold values
    drowsiness_label = "Open Eyes" if drowsiness_prediction[0] > 0.5 else "Closed Eyes"
    yawn_label = "Yawn" if yawn_prediction[0] > 0.5 else "No Yawn"

    # Display the frame and predictions
    cv2.putText(frame_bgr, f"Drowsiness: {drowsiness_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame_bgr, f"Yawn: {yawn_label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Webcam', frame_bgr)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
