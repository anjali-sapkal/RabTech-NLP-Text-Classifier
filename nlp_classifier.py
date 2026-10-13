
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load dataset
url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
df = pd.read_csv(url, sep="\t", header=None,
                 names=["label", "message"])

df["label"] = df["label"].map({"ham": 0, "spam": 1})
df = df.dropna()

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["message"], df["label"],
    test_size=0.2, random_state=42, stratify=df["label"]
)

# 2. TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
X_train = vectorizer.fit_transform(X_train_text).toarray()
X_test = vectorizer.transform(X_test_text).toarray()

# 3. Build neural network
model = Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)),
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.4),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

# 4. Train with early stopping
early_stop = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=10, batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop], verbose=1
)

# 5. Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("\nTest accuracy:", round(accuracy * 100, 2), "%")

predictions = (model.predict(X_test, verbose=0).ravel() >= 0.5).astype(int)
print(classification_report(
    y_test, predictions, target_names=["Not Spam", "Spam"]
))

# 6. Plot training curves
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()

# 7. Test sample messages
samples = [
    "Congratulations! You have won a free prize. Claim now!",
    "Hi, are we meeting for class tomorrow?"
]

sample_vectors = vectorizer.transform(samples).toarray()
probs = model.predict(sample_vectors, verbose=0).ravel()

for message, prob in zip(samples, probs):
    label = "SPAM" if prob >= 0.5 else "NOT SPAM"
    confidence = prob if prob >= 0.5 else 1 - prob
    print("\nMessage:", message)
    print("Prediction:", label)
    print("Confidence:", round(float(confidence) * 100, 2), "%")

# 8. Save model
model.save("spam_text_classifier.keras")
print("\nSaved model: spam_text_classifier.keras")
