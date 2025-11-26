import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator


input_shape = (256, 256, 3)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

from keras.optimizers import Adam

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

train_dir = 'data/dataset/train'
validation_dir = 'data/dataset/validation'

train_datagen = ImageDataGenerator(rescale=1./255,
                                   #rotation_range=20,
                                   #width_shift_range=0.1,
                                   #height_shift_range=0.1,
                                   #horizontal_flip=True,
                                   #fill_mode='nearest'
                                   )


validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary'
)

from keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint_file_path = 'best_model_EPOCH100.h5'

checkpoint_callbacks = ModelCheckpoint(filepath=checkpoint_file_path, save_best_only=True, monitor='val_accuracy', mode='max')
early_stopping_callbacks = EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss', mode='min')

history = model.fit(
    train_generator,
    epochs=100,
    validation_data=validation_generator,
    callbacks=[checkpoint_callbacks, early_stopping_callbacks]
)

def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label="Training Loss")
    plt.plot(history.history['val_loss'], label="Validation Loss")
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('train_validation_accuracy.png')

plot_training_history(history)