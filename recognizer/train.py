import tensorflow as tf
import numpy as np
from tensorflow.keras import callbacks
from .data_preparation import get_preprocessed_data
from config import TRAINING_EPOCH, BATCH_SIZE,LOSS_WEIGHTS, LEARNING_RATE, VAL_SPLIT, PRETRAINED_DIR


def compile_model(model):
    ''''''
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=LEARNING_RATE),
                  loss={'classification':tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                        'reconstruction':tf.keras.losses.MeanSquaredError()},
                  metrics=['accuracy'],
                  loss_weights = LOSS_WEIGHTS)

    return model

def do_train(model):
    model = compile_model(model)

    if PRETRAINED_DIR !='':
        model.load_weights(PRETRAINED_DIR)

    (x_train, y_train,y_train_synthetic), (_, _,_) = get_preprocessed_data()

    shuffle_index = np.random.permutation(len(y_train))

    x_train, y_train,y_train_synthetic = x_train[shuffle_index], y_train[shuffle_index],y_train_synthetic[shuffle_index]

    partition = int(VAL_SPLIT*len(y_train))

    #Tensorboad & Saving Best Model
    my_callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./data/saved_model/model.{epoch:02d}-{val_reconstruction_accuracy:.5f}.h5',\
                         monitor="val_reconstruction_accuracy",mode='max',save_best_only=True),
                    tf.keras.callbacks.TensorBoard(log_dir='./data/logs'),
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')]


    history = model.fit(x_train[:partition],[y_train_synthetic[:partition],y_train[:partition]], \
        batch_size = BATCH_SIZE,epochs = TRAINING_EPOCH,callbacks = my_callbacks,
        validation_data=(x_train[partition:],[y_train_synthetic[partition:],y_train[partition:]]))

    return model, history