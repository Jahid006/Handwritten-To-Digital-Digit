import tensorflow as tf
from .spatial_transformation import Localization, BilinearInterpolation

def get_model(input_shape=(28,28,1)):
    ''' Return Model'''
    image = tf.keras.layers.Input(shape=input_shape)
    theta = Localization()(image)
    x = BilinearInterpolation(height=input_shape[0], width=input_shape[1])([image, theta])
    x = tf.keras.layers.Conv2D(64, [3, 3],padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(64, [3, 3],padding='same', activation='relu')(x)
    xy = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(128, [3, 3],padding='same', activation='relu')(xy)
    x = tf.keras.layers.Conv2D(64, [1, 1],padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(64,[2, 2],strides=2,padding='valid', activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, [1, 1],padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(64,[2, 2],strides=2,padding='valid', activation='relu')(x)
    x = tf.keras.layers.Conv2D(32, [3, 3],padding='same', activation='relu',name = 'second_last')(x)
    x = tf.keras.layers.Conv2D(1, [1, 1],padding='same', activation=None,name ='reconstruction')(x)

    z = tf.keras.layers.Flatten()(xy)
    z = tf.keras.layers.Dense(64, activation='relu')(z)
    z = tf.keras.layers.Dense(32, activation='relu')(z)
    z = tf.keras.layers.Dense(10, activation='softmax',name='classification')(z)

    return tf.keras.models.Model(inputs=image, outputs=[x,z])