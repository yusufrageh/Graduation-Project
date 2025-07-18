
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Conv2D, Dropout,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import metrics
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import albumentations as A
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


model = models.Sequential()

# First convolutional layer
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(160, 160, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Second convolutional layer
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third convolutional layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fourth convolutional layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fifth convolutional layer
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


# Flatten the feature maps
model.add(Flatten())

# Fully connected layer
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))  # Dropout for regularization

# Output layer
model.add(Dense(1, activation='sigmoid'))


# Display the model summary
model.summary()


# Compile the model
model.compile(
    optimizer=Adam(lr=0.00002),  # Use Adam optimizer with an initial learning rate
    loss='binary_crossentropy',
    metrics=[
        metrics.BinaryAccuracy(),
        metrics.Precision(),
        metrics.Recall(),
        metrics.AUC()
    ]
)
# Display the model summary
model.summary()

# Define data augmentation techniques using Albumentations
augmentation_transform = A.Compose([
    A.HorizontalFlip(p=0.3),
], p=0.5)

# Define a function to apply augmentation to images
def augment_images(image):    return augmentation_transform(image=image)['image']
    
# Update ImageDataGenerator with preprocessing for MobileNetV2 and data augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=augment_images,  # Apply data augmentation
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(160, 160),
    batch_size=128,
    class_mode='binary',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(160, 160),
    batch_size=128,
    class_mode='binary',
    subset='validation',
    shuffle=True
)

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
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
    steps_per_epoch=train_generator.samples // 128,
    epochs=150,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 128,
    callbacks=[early_stopping, lr_callback]
)

# Plot metrics
plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 4, 1)
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('binary_accuracy')
plt.ylabel('binary_accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Loss
plt.subplot(1, 4, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Precision
plt.subplot(1, 4, 3)
plt.plot(history.history['precision'])
plt.plot(history.history['val_precision'])
plt.title('Precision')
plt.ylabel('Precision')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Recall
plt.subplot(1, 4, 4)
plt.plot(history.history['recall'])
plt.plot(history.history['val_recall'])
plt.title('Recall')
plt.ylabel('Recall')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()
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
model.save('trial.h5')
