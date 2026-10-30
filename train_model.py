"""
train_model.py

Trains a food item classifier using transfer learning on MobileNetV2.
Run this AFTER split_dataset.py has produced dataset_split/{train,val,test}.

Usage:
    python train_model.py

Outputs:
    model/food_classifier.h5      - trained Keras model
    model/class_names.json        - list of class names (index -> label)
    training_history.png          - accuracy/loss curves
"""

import os
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

DATA_DIR = "dataset_split"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 5
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)


def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "val"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "test"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    class_names = train_ds.class_names
    print(f"Classes ({len(class_names)}): {class_names}")

    # Preprocess for MobileNetV2 (scales pixels to [-1, 1])
    def preprocess(x, y):
        return preprocess_input(x), y

    train_ds = train_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def build_model(num_classes):
    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.1),
    ])

    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # freeze for initial training

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base_model


def plot_history(history, fine_tune_history=None, out_path="training_history.png"):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    if fine_tune_history:
        acc += fine_tune_history.history["accuracy"]
        val_acc += fine_tune_history.history["val_accuracy"]
        loss += fine_tune_history.history["loss"]
        val_loss += fine_tune_history.history["val_loss"]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Train Accuracy")
    plt.plot(val_acc, label="Val Accuracy")
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.legend()
    plt.title("Loss")

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved training curves to {out_path}")


def main():
    train_ds, val_ds, test_ds, class_names = load_datasets()
    num_classes = len(class_names)

    model, base_model = build_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    print("\n--- Phase 1: Training top layers (base frozen) ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
    )

    print("\n--- Phase 2: Fine-tuning top of base model ---")
    base_model.trainable = True
    # Only unfreeze the last ~30 layers to avoid overfitting on a small dataset
    fine_tune_at = len(base_model.layers) - 30
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    fine_tune_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
    )

    print("\n--- Evaluating on test set ---")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # Save model and class names
    model_path = os.path.join(MODEL_DIR, "food_classifier.h5")
    model.save(model_path)
    print(f"Saved model to {model_path}")

    class_names_path = os.path.join(MODEL_DIR, "class_names.json")
    with open(class_names_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Saved class names to {class_names_path}")

    plot_history(history, fine_tune_history)

    # Write a small results summary for the report
    with open(os.path.join(MODEL_DIR, "results_summary.txt"), "w") as f:
        f.write(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}\n")
        f.write(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}\n")
        f.write(f"Test accuracy: {test_acc:.4f}\n")
        f.write(f"Test loss: {test_loss:.4f}\n")
        f.write(f"Classes: {class_names}\n")

    print("\nDone. Use model/results_summary.txt for your report's accuracy section.")


if __name__ == "__main__":
    main()
