import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import metrics
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import albumentations as A
import cv2
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load the pre-trained MobileNetV2 model without the top (fully connected) layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(160, 160, 3))

# Fine-tune the last 20 layers
for layer in base_model.layers[:-5]:
    layer.trainable = False
for layer in base_model.layers[-5:]:
    layer.trainable = True

# Add custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(16, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(32, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(
    optimizer=Adam(lr=0.0001),  # Use Adam optimizer with an initial learning rate
    loss='binary_crossentropy',
    metrics=[
        metrics.BinaryAccuracy(),
        metrics.Precision(),
        metrics.Recall(),
        metrics.AUC()
    ]
)

# Define data augmentation techniques using Albumentations
augmentation_transform = A.Compose([
    A.HorizontalFlip(p=0.3),
    A.ShiftScaleRotate(p=0.3),
    #A.RandomBrightnessContrast(p=0.2)
], p=0.5)

# Define a function to apply augmentation to images
def augment_images(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = np.stack((image,)*3, axis=-1)  # Convert to 3 channels
    return augmentation_transform(image=image)['image']

# Update ImageDataGenerator with preprocessing for MobileNetV2 and data augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=augment_images,  # Apply data augmentation
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(160, 160),
    batch_size=64,
    class_mode='binary',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(160, 160),
    batch_size=64,
    class_mode='binary',
    subset='validation',
    shuffle=True
)

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=4,
    restore_best_weights=True
)

# Learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch % 4 == 0:
        return lr * 0.9  # Decrease learning rate by 10% every 4 epochs
    else:
        return lr

lr_callback = LearningRateScheduler(lr_scheduler)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 64,
    epochs=150,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 64,
    callbacks=[early_stopping, lr_callback]
)

# Evaluate the model
predicted_labels = []
true_labels = []
for i in range(len(validation_generator)):
    batch_images, batch_labels = validation_generator.next()
    predicted_labels.extend((model.predict(batch_images) > 0.5).astype(int))
    true_labels.extend(batch_labels)

predicted_labels = [item.flatten()[0] for item in predicted_labels]
true_labels = [int(item.flatten()[0]) for item in true_labels]

# Calculate F1-score and Confusion Matrix on the validation set
f1 = f1_score(true_labels, predicted_labels)
print(f'F1-score on validation set: {f1}')

conf_matrix_normalized = confusion_matrix(true_labels, predicted_labels, normalize='true')

# Plot the confusion matrix
plt.figure(figsize=(6, 6))
class_names = list(train_generator.class_indices.keys())
sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2%', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Normalized Confusion Matrix')
plt.show()

# Save the fine-tuned MobileNetV2 model
model.save('pretrained_mobilenetv2_augmented_gray_2.h5')
