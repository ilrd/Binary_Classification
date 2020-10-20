import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from sklearn.model_selection import KFold
from tensorflow.keras import Model
from sklearn.preprocessing import Normalizer
from tensorflow.keras import callbacks
import datetime

train_df = pd.read_csv('../../data/train_processed.csv')
X_test = pd.read_csv('../../data/test_processed.csv')

# Shuffling
train_df = train_df.sample(frac=1).reset_index(drop=True)

label = 'Survived'

X_train = train_df.drop(columns=[label])
y_train = train_df[label]


def get_callbacks():
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_filepath = f'checkpoints/model_checkpoint.h5'

    fit_callbacks = [
        callbacks.ReduceLROnPlateau(
            monitor='val_acc',
            factor=0.1,
            patience=15,
            cooldown=10,
            min_lr=1e-5,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_acc',
            verbose=1,
            save_best_only=True,
        ),
        callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
        ),

        callbacks.EarlyStopping(
            monitor='val_acc',
            patience=30,
        ),
    ]

    return fit_callbacks


def build_model():
    inputs = Input(shape=X_train[0].shape)
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)

    model.compile('adam', 'binary_crossentropy', metrics='acc')

    return model


# Normalizing
scaler = Normalizer()

X_train['Fare'] = scaler.fit_transform([X_train['Fare']])[0]
X_train['Age'] = scaler.fit_transform([X_train['Age']])[0]

X_test['Fare'] = scaler.transform([X_test['Fare']])[0]
X_test['Age'] = scaler.transform([X_test['Age']])[0]

# Turning Dataframe to ndarray
X_train = X_train.values
X_test = X_test.values

# Callbacks
fit_callbacks = get_callbacks()

# K-fold cross validation
K_SPLITS = 10
kf = KFold(K_SPLITS)

ACCURACIES = []
for train_indecies, val_indecies in kf.split(X_train, y_train):
    X_fold_train = X_train[train_indecies]
    X_fold_val = X_train[val_indecies]

    y_fold_train = y_train[train_indecies]
    y_fold_val = y_train[val_indecies]

    model = build_model()

    history = model.fit(X_fold_train, y_fold_train, epochs=200, validation_data=(X_fold_val, y_fold_val),
                        callbacks=fit_callbacks)

    loss, acc = model.evaluate(X_fold_val, y_fold_val)

    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(history.history['acc'], label='acc')
    plt.plot(history.history['val_acc'], label='val_acc')
    plt.legend()
    plt.show()

    ACCURACIES.append(acc)

print(np.array(ACCURACIES).mean())  # Gives 0.7979775190353393


