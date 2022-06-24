from tensorflow.keras.datasets import mnist
import numpy as np
from .synthatic_label import get_label

def get_preprocessed_data(ifTrain = True):
    ''' Return Normalized Train & Test Image Data as Numpy'''

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    synthetic_labels = get_label(10)

    y_train_synthetic = np.array([np.array(synthetic_labels[i]).reshape(28,28,1)/255 for i in y_train])
    y_test_synthetic = np.array([np.array(synthetic_labels[i]).reshape(28,28,1)/255 for i in y_test])

    if ifTrain:
        return (x_train/255, y_train,y_train_synthetic), (x_test/255, y_test,y_test_synthetic)

    if not ifTrain:
        return x_test/255, y_test,y_test_synthetic