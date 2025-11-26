import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import random

MODEL_PATH = 'best_model_EPOCH100.h5'

BASE_DIR = 'data/dataset'
TEST_DIR = os.path.join(BASE_DIR, 'test')

IMG_SIZE = (256, 256)
BATCH_SIZE = 32
NUM_IMAGE_TO_SHOW = 10

print("load best model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("model load complete")

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    class_mode='binary',
    shuffle=False
)

print("test model with test dataset")
loss, accuracy = model.evaluate(test_generator)
print(f"[Loss]: {loss:.4f}")
print(f"[Accuracy]: {accuracy:.4f}")

print("prediction with test dataset")
predictions = model.predict(test_generator)

y_pred = (predictions > 0.5).astype("int32").flatten()
y_true = test_generator.classes
filenames = test_generator.filenames

labels_map = {v: k for k, v in test_generator.class_indices.items()}

correct_indices = []
incorrect_indices = []
for i in range(len(y_pred)):
    if y_pred[i] == y_true[i]:
        correct_indices.append(i)
    else:
        incorrect_indices.append(i)

def visualize_prediction(indices, title):
    sample_indices = random.sample(indices, min(NUM_IMAGE_TO_SHOW, len(indices)))

    if not sample_indices:
        print(f"\n{title} image not found")
        return
    
    num_cols = 5
    num_rows = 2

    plt.figure(figsize=(num_cols * 3, num_rows * 3.5))
    plt.suptitle(title, fontsize=16)

    for i, idx in enumerate(sample_indices):
        plt.subplot(num_rows, num_cols, i + 1)

        img_path = os.path.join(TEST_DIR, filenames[idx])
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)

        true_label = labels_map[y_true[idx]]
        pred_label = labels_map[y_pred[idx]]

        subplot_title = f"Real : {true_label}\n Pred: {pred_label}"

        plt.imshow(img)
        plt.title(subplot_title, color='green' if true_label == pred_label else 'red')
        plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'true_and_pred_{subplot_title}.png')



class_names = list(test_generator.class_indices.keys())

print("Classification Report")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Prediced Label', fontsize=12)
plt.ylabel('Actual Label', fontsize=12)
plt.savefig('Confusion Matrix')

visualize_prediction(correct_indices, "Correct Predictions")
visualize_prediction(incorrect_indices, "Incorrect Predictions")
