import numpy as np
import cv2
import glob
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import Sequential, layers
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


# DATA
data_list = []
label_list = []

for i, address in enumerate(glob.glob("../session17/fire_dataset/*/*")):
    image = cv2.imread(address)
    if image is None:
        continue
    image = cv2.resize(image, (32, 32))
    image = image / 255
    image = image.flatten()

    data_list.append(image)
    label_list.append(address.replace("\\", "/").split("/")[-2])

    if i % 200 == 0:
        print(f'[INFO] {i} images processed!')


X = np.array(data_list)
y = np.array(label_list)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# MODEL
model = Sequential([
    layers.Dense(20, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(X_train, y_train, validation_data=(
    X_test, y_test), batch_size=128, epochs=10)

model.save("fire_detector_nn.h5")
