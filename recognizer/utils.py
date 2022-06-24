from operator import pos
import numpy as np
import PIL, os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import POSTPROCESS, FIGURES_DIR


def showImage(pred,ground_truth = None, show_image = True, save_image = False,output_dir = None, imname = None, concate = False):

    if POSTPROCESS: pred = postProcess(pred)
    if concate: pred = np.concatenate((ground_truth.reshape(28,28),pred.reshape(28,28)),axis=1)
    
    image = Image.fromarray(pred).convert("L")

    if show_image:image.show()
    if save_image:
        os.makedirs(output_dir,exist_ok=True)
        image.save(os.path.join(output_dir,imname))


def generate_report(prediction, y_test,x_test,output_dir = None, show_image= False, save_image = False):

    classification = np.argmax(prediction[1],axis=1).reshape(-1)
    classification = classification.reshape(classification.shape[0])

    accuracy = 100*np.sum(classification==y_test)/len(y_test)
    images = prediction[0]*255
    x_test = x_test*255

    print(f"\n\n--------Saving Images to {output_dir}--------\n\n")
    if save_image:
        for i in tqdm(range(len(y_test))):
            imname =  str(classification[i])+'_'+str(y_test[i])+"_"+str(i)+'.jpg'
            showImage(images[i], x_test[i],show_image = show_image,save_image = save_image, output_dir = output_dir,imname = imname,concate=True )

def postProcess(image):
    mean = np.mean(image[image>4])*.33

    image[image<mean] = 0
    image[image>mean] = 255

    return image

def plot_result(train,test):
    name = ['reconstruction_loss','classification_loss',  'reconstruction_accuracy','classification_accuracy']
    train = train[1:]
    test = test[1:]
    os.makedirs('figures',exist_ok=True)

    metrices = {}
    print('\n\n----------------------Performance Metrics--------------------\n')
    for i in range(len(train)):
        metrices[name[i]] = [train[i],test[i]]

        fig = plt.figure(figsize = (7, 7))
        plt.bar(['Train','Test'],metrices[name[i]],width=.4)

        plt.xlabel("Data Partition")
        plt.ylabel(name[i])
        plt.title(f"{name[i].upper()} in different Data Partiton ")

        plt.savefig(os.path.join(FIGURES_DIR,str(name[i])+'.jpg'))

        print(f" {name[i].upper()}: Train: {train[i]}, Test: {test[i]}")










