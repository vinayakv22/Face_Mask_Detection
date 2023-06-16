# %% [markdown]
# # ResNet 50 Implementation

# %% [markdown]
# ## Importing Libraries

# %%
import os
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import numpy as np

# %% [markdown]
# ## Defining Model Parameters

# %%
train_path = "fmd dataset/Train"
test_path = "fmd dataset/Test"
val_path = "fmd dataset/Validation"

seed = 0
image_size = 100
channels = 3
validation_split = 0.2
batch_size = 32
num_epochs = 30
learning_rate = 0.00001
early_stop_patience = 10
classes = ['WithMask', 'WithoutMask']

# %% [markdown]
# ## Retrieving training, validation and test datasets

# %%
train = image_dataset_from_directory(
    directory=train_path,
    labels="inferred",
    class_names=classes,
    label_mode="int",
    batch_size=batch_size,
    image_size=(image_size, image_size),
    shuffle=True
)

validation = image_dataset_from_directory(
    directory=val_path,
    labels="inferred",
    class_names=classes,
    label_mode="int",
    batch_size=batch_size,
    image_size=(image_size, image_size),
    shuffle=True
)

test = image_dataset_from_directory(
    directory=test_path,
    labels="inferred",
    class_names=classes,
    label_mode="int",
    batch_size=batch_size,
    image_size=(image_size, image_size),
    shuffle=False
)

# %% [markdown]
# ## Model Architecture

# %%
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# %%
input_tensor = tf.keras.layers.Input(shape=(image_size, image_size, channels))
model = tf.keras.applications.resnet50.ResNet50(
    weights='imagenet',
    input_tensor=input_tensor,
    include_top=False,
    pooling='avg',
    classes=2,
)

# Apply CNN Headers on top of the base model
xin = model.output
xout = Dense(512, activation="relu")(xin)
xout = Dropout(0.2)(xout)
xout = Dense(256, activation="relu")(xout)
xout = Dropout(0.2)(xout)
xout = Dense(128, activation="relu")(xout)
xout = Dropout(0.2)(xout)
xout = Dense(64, activation="relu")(xout)
xout = Dropout(0.2)(xout)
xout = Dense(32, activation="relu")(xout)
xout = Dropout(0.2)(xout)
xout = Dense(16, activation="relu")(xout)
xout = Dropout(0.2)(xout)
xout = Dense(8, activation="relu")(xout)
xout = Dropout(0.2)(xout)
xout = Dense(4, activation="relu")(xout)
xout = Dropout(0.2)(xout)
xout = Dense(2, activation="relu")(xout)
xout = Dropout(0.2)(xout)
xout = Dense(len(classes), activation='softmax')(xout)

detector = Model(inputs=model.input, outputs=xout)

early_stopping = EarlyStopping( monitor="val_loss", mode="min", verbose=1, restore_best_weights=True)

# %%
model.summary()

# %% [markdown]
# ## Checking Cudas

# %%
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# %% [markdown]
# ## Training on Cudas if available

# %%
optimizer = Adam(learning_rate=learning_rate)
loss = SparseCategoricalCrossentropy()
metrics = [SparseCategoricalAccuracy()]

# %%
with tf.device('/GPU:0'):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    dmodel = model.fit(train, validation_data=validation, epochs=num_epochs)

# %% [markdown]
# ## Plotting training progress

# %%
loss_curve = pd.DataFrame(dmodel.history)

loss_curve[["sparse_categorical_accuracy", "val_sparse_categorical_accuracy"]].plot()
loss_curve[["loss", "val_loss"]].plot()

# %% [markdown]
# ## Predict from the above trained model

# %%
y_true = []
y_pred = []

for x_test_batch, y_test_batch in test:
    y_true.append(y_test_batch)
    predictions = detector.predict(x_test_batch, verbose=0)
    y_pred.append(np.argmax(predictions, axis=1))

y_true = tf.concat(y_true, axis=0)
y_pred = tf.concat(y_pred, axis=0)

# %% [markdown]
# ## Evaluation of Model

# %%
print(classification_report(y_true, y_pred, target_names=classes, digits=4))

# %%
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# %% [markdown]
# ## Optimizing the model for Embedded Systems and Mobile Phones using TensorFlow Lite

# %%
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
with open('ResNet50.tflite', 'wb') as f:
    f.write(tflite_model)

# %%
# Use the model
interpreter = tf.lite.Interpreter(model_path='ResNet50.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)


