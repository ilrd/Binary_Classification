import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from sklearn.model_selection import KFold
from tensorflow.keras import Model
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import callbacks
import datetime

train_df = pd.read_csv('../../data/train_processed.csv')
X_test = pd.read_csv('../../data/test_processed.csv')

# Shuffling
train_df = train_df.sample(frac=1).reset_index(drop=True)

label = 'Survived'

X_train = train_df.drop(columns=[label])
y_train = train_df[label]


def get_callbacks(val=True):
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if val:
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
                patience=50,
            ),
        ]

    else:
        checkpoint_filepath = f'checkpoints/best_model.h5'
        fit_callbacks = [
            callbacks.ReduceLROnPlateau(
                monitor='acc',
                factor=0.1,
                patience=15,
                cooldown=10,
                min_lr=1e-5,
                verbose=1,
            ),
            callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                monitor='acc',
                verbose=1,
                save_best_only=True,
            ),
            callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
            ),
            callbacks.EarlyStopping(
                monitor='acc',
                patience=50,
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


# Standardizing
scaler = StandardScaler()

X_train[['Fare', 'Age']] = scaler.fit_transform(X_train[['Fare', 'Age']])
X_test[['Fare', 'Age']] = scaler.transform(X_test[['Fare', 'Age']])

# Turning Dataframe to ndarray
X_train = X_train.values
X_test = X_test.values


def train_model():
    # Callbacks
    fit_callbacks = get_callbacks(val=True)

    # K-fold cross validation
    K_SPLITS = 10
    kf = KFold(K_SPLITS)

    ACCURACIES = []
    for train_indices, val_indices in kf.split(X_train, y_train):
        X_fold_train = X_train[train_indices]
        X_fold_val = X_train[val_indices]

        y_fold_train = y_train[train_indices]
        y_fold_val = y_train[val_indices]

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

    print(np.array(ACCURACIES).mean())  # Gives ~0.83


fit_callbacks = get_callbacks(val=False)
best_model = build_model()
best_history = best_model.fit(X_train, y_train, 32, epochs=70, callbacks=fit_callbacks)

# To save the prediction on testing set
best_model = keras.models.load_model('checkpoints/best_model.h5')
y_pred = np.round(best_model.predict(X_test).flatten()).astype(int)

submission_df = pd.DataFrame(columns=['PassengerId', 'Survived'], data=zip(np.arange(892, 1310), y_pred))
submission_df.to_csv('submission.csv', index=False)
