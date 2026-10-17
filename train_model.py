
from pathlib import Path
import tensorflow as tf

PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT_DIR / "Training"
TEST_DIR = PROJECT_DIR / "Testing"
MODEL_PATH = PROJECT_DIR / "brain_tumor_model.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 5
SEED = 42

if not TRAIN_DIR.exists() or not TEST_DIR.exists():
    raise FileNotFoundError(
        "Training or Testing folder was not found in the project directory."
    )

classes = ["glioma", "meningioma", "notumor", "pituitary"]

for folder in [TRAIN_DIR, TEST_DIR]:
    missing = [name for name in classes if not (folder / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Missing class folders in {folder.name}: {missing}"
        )

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    class_names=classes,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    validation_split=0.15,
    subset="training",
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    class_names=classes,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    validation_split=0.15,
    subset="validation",
    seed=SEED
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    class_names=classes,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

base_model = tf.keras.applications.Xception(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
    pooling="avg"
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.applications.xception.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(4, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        str(MODEL_PATH),
        monitor="val_accuracy",
        save_best_only=True
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )
]

print("Class order:", classes)
print("Starting training...")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("\nEvaluating on the Testing dataset...")
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print(f"Test loss: {test_loss:.4f}")

model.save(MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")
