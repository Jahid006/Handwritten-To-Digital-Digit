import tensorflow as tf
import numpy as np
from PIL import Image
import glob, os
import numpy as np

from config import LAYER_VISUALIZATION_DIR

def visualize(model,img):
    for layer in model.layers:
        layer.trainable = False

    v_model = tf.keras.models.Model(model.inputs,model.get_layer('second_last').output)

    #print(v_model.summary())

    preds = v_model.predict(img.reshape(1,28,28,1))[0]

    os.makedirs(LAYER_VISUALIZATION_DIR,exist_ok=True)

    for i in range(preds.shape[2]):
        temp = Image.fromarray(preds[:,:,i]*255).convert('RGB')
        temp.save(os.path.join(LAYER_VISUALIZATION_DIR,str(i)+'.jpg'))

