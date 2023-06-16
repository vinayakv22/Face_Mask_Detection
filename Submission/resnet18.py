# %% [markdown]
# # ResNet18 Implementation

# %% [markdown]
# ## Importing Libraries

# %%
import tensorflow as tf
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.layers import Layer
from typing import Tuple, Union
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import resnets as rn
import pandas as pd
import numpy as np
import os

# %% [markdown]
# ## Set hyperparameters

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

# %%
classes = next(os.walk(train_path))[1]
print(classes)

# %% [markdown]
# ## Create training, validation and test datasets

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
# ## ResNet-18 Architecture (Utilizing resnets Library)

# %%
def ResNet18(
    input_shape: Tuple[int, int, int],
    output_units=1000,
    include_top=True,
    after_input: Union[Sequential, Layer, None] = None,
    normalize=False,
    kernel_regularizer: Union[Regularizer, None] = None,
    kernel_initializer="he_uniform",
    flatten=False,
    dropout_rate=0.0,
) -> Model:

    return rn.ResNet(
        input_shape,
        (2, 2, 2, 2),
        "small",
        output_units=output_units,
        include_top=include_top,
        after_input=after_input,
        normalize=normalize,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer,
        flatten=flatten,
        dropout_rate=dropout_rate,
    )

# %% [markdown]
# ## Custom architecture with Dense and Dropout Layers

# %%
optimizer = Adam(learning_rate=learning_rate)
loss = SparseCategoricalCrossentropy()
metrics = [SparseCategoricalAccuracy()]

model = ResNet18(
    (image_size, image_size, channels),
    include_top=False,
    normalize=True,
    flatten=True,
    dropout_rate=0.2
)

# Apply CNN Headers on top of the base model
xin = model.output
xout = Dense(4096, activation='relu')(xin)
xout = Dropout(0.5)(xout)
xout = Dense(2048, activation='relu')(xout)
xout = Dropout(0.5)(xout)
xout = Dense(1024, activation='relu')(xout)
xout = Dropout(0.5)(xout)
xout = Dense(512, activation='relu')(xout)
xout = Dropout(0.5)(xout)
xout = Dense(256, activation='relu')(xout)
xout = Dropout(0.5)(xout)
xout = Dense(128, activation='relu')(xout)
xout = Dropout(0.5)(xout)
xout = Dense(64, activation='relu')(xout)
xout = Dropout(0.5)(xout)
xout = Dense(len(classes), activation='softmax')(xout)

# Combine the base model and the CNN headers
detector = Model(inputs=model.input, outputs=xout)

early_stopping = EarlyStopping(
    monitor="val_loss",
    mode="min",
    verbose=1,
    patience=early_stop_patience,
    restore_best_weights=True
)

# %%
detector.summary()

# %% [markdown]
# ## Checking Cudas

# %%
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# %% [markdown]
# ## Training on Cudas if available

# %%
# moving tensorflow to GPU
with tf.device('/GPU:0'):
    detector.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    dmodel = detector.fit(train, validation_data=validation, epochs=num_epochs)

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

with tf.device('/GPU:0'):
    for x_test_batch, y_test_batch in test:
        y_true.append(y_test_batch)
        predictions = detector.predict(x_test_batch, verbose=0)
        y_pred.append(np.argmax(predictions, axis=1))

y_true = tf.concat(y_true, axis=0)
y_pred = tf.concat(y_pred, axis=0)

# %% [markdown]
# ## Visualizing the Results

# %%
# Visualize the results on test dataset for some images
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(classes[y_pred[i]])
    plt.axis("off")

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
with open('Resnet18.tflite', 'wb') as f:
    f.write(tflite_model)

# %%
# Use the model
interpreter = tf.lite.Interpreter(model_path='Resnet18.tflite')
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


