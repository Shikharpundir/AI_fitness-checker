import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Conv1D, BatchNormalization, Activation, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--arch", choices=["lstm", "tcn"], default="lstm")
parser.add_argument("--epochs", type=int, default=80)
args = parser.parse_args()

X = np.load("data/processed_sequences/X_balanced.npy")
y_raw =np.load("data/processed_sequences/y_balanced.npy")

# Angles are bounded in [0, 180]. Matching this normalization in live
# inference makes the model less sensitive to feature scale and initialization.
X = X / 180.0

encoder = LabelEncoder()
y_int = encoder.fit_transform(y_raw)
y = to_categorical(y_int)
num_classes = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y_int, random_state=42
)

class_values = np.unique(y_int)
weights = compute_class_weight(
    class_weight="balanced", classes=class_values, y=y_int
)
class_weight = dict(zip(class_values.tolist(), weights.tolist()))


def build_lstm(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(96, return_sequences=True),
        Dropout(0.3),
        LSTM(48),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    return model


def build_tcn(input_shape, num_classes):
    model = Sequential([Input(shape=input_shape)])
    dilations = [1, 2, 4, 8]
    for i, d in enumerate(dilations):
        model.add(Conv1D(64, kernel_size=3, dilation_rate=d, padding="causal"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.2))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(32, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))
    return model


input_shape = (X.shape[1], X.shape[2])  # (30, 8)
model = build_lstm(input_shape, num_classes) if args.arch == "lstm" else build_tcn(input_shape, num_classes)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

os.makedirs("models", exist_ok=True)
model_path = f"models/exercise_{args.arch}.h5"
early_stop = EarlyStopping(monitor="val_accuracy", patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5, verbose=1
)
checkpoint = ModelCheckpoint(
    model_path, monitor="val_accuracy", save_best_only=True, mode="max", verbose=0
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=args.epochs,
    batch_size=8,
    class_weight=class_weight,
    callbacks=[early_stop, reduce_lr, checkpoint],
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal test accuracy: {test_acc:.4f}")

model.save(model_path)
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print(f"Saved model to {model_path}")