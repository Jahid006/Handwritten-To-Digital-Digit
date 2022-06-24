import tensorflow as tf
import numpy as np
from PIL import Image
import glob


from recognizer.visualize_layer import visualize
from recognizer.utils import generate_report, plot_result, showImage
from recognizer.data_preparation import get_preprocessed_data
from recognizer.model import get_model
from recognizer.train import do_train, compile_model
from config import EVALUATE_PERFORMANCE, PRINT_SUMMARY,TRAINING,PRETRAINED_DIR, TESTING, OUTPUT_DIRS, DEMO, DEMO_DIR,\
     VISULIZE_2ND_LAST_LAYER, INFERENCE_BATCH_SIZE, NO_OF_IMAGE_FOR_VISUALIZATION



def training():
    model_ = get_model()
    if PRINT_SUMMARY: model_.summary()

    model_, history = do_train(model_)

def test():

    x_test, y_test,y_test_synthetic = get_preprocessed_data(ifTrain = False)
    shuffle_index = np.random.permutation(len(y_test))
    x_test, y_test,y_test_synthetic = x_test[shuffle_index], y_test[shuffle_index], y_test_synthetic[shuffle_index]

    model_ = get_model()
    model_ = compile_model(model_)
    model_.load_weights(PRETRAINED_DIR)

    report = model_.evaluate(x_test,[y_test_synthetic,y_test],batch_size=INFERENCE_BATCH_SIZE,verbose=1)

    #Generates 256 [images,Output] pair in the given directory
    if NO_OF_IMAGE_FOR_VISUALIZATION>0:
        prediction = model_.predict(x_test[:NO_OF_IMAGE_FOR_VISUALIZATION])
        generate_report(prediction,y_test[:NO_OF_IMAGE_FOR_VISUALIZATION],x_test[:NO_OF_IMAGE_FOR_VISUALIZATION],\
            OUTPUT_DIRS, show_image = False, save_image = True)

def demo(img_dir):
    files = glob.glob(img_dir+'*.png')

    image_files = []

    for i in files:
        img = Image.open(i).convert('L').resize((28,28))
        img = np.array(img)/255
        image_files.append(img)

    image_files = np.array(image_files).reshape(len(image_files),28,28,1)

    model_ = get_model()
    model_ = compile_model(model_)
    model_.load_weights(PRETRAINED_DIR)

    prediction = model_.predict(image_files)

    for k,i in enumerate(prediction[0]):
        yinput = image_files[k].reshape(28,28)*255
        yinput = yinput.astype(int)
        showImage(i.reshape(28,28)*255,yinput,show_image= True,save_image = False, imname=None,concate=True)

def evaluate_performance():
    (x_train, y_train,y_train_synthetic), (x_test, y_test,y_test_synthetic) = get_preprocessed_data()
    model_ = get_model()
    model_ = compile_model(model_)
    model_.load_weights(PRETRAINED_DIR)

    report_test  = model_.evaluate(x_test,  [y_test_synthetic,y_test], batch_size=INFERENCE_BATCH_SIZE,verbose=0)
    report_train = model_.evaluate(x_train,[y_train_synthetic,y_train],batch_size=INFERENCE_BATCH_SIZE,verbose=0)
    plot_result(report_train,report_test)

def visualization_adapter(img_dir):

    files = glob.glob(img_dir+'*.png')[0]
    img = Image.open(files).convert('L').resize((28,28))
    img = np.array(img)/255


    model_ = get_model()
    model_ = compile_model(model_)

    #print(model_.summary())
    model_.load_weights(PRETRAINED_DIR)

    visualize(model_,img)

def main():
    if TRAINING:
        training()
    if TESTING  and not TRAINING:
        test()
    if DEMO:
        demo(DEMO_DIR)

    if VISULIZE_2ND_LAST_LAYER:
        visualization_adapter(DEMO_DIR)
    if EVALUATE_PERFORMANCE:
        evaluate_performance()

if __name__ == '__main__':
    main()


