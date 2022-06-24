TRAINING = False
TESTING = False
DEMO = False 
EVALUATE_PERFORMANCE = False
VISULIZE_2ND_LAST_LAYER = True 

POSTPROCESS = False
PRINT_SUMMARY = False

NO_OF_IMAGE_FOR_VISUALIZATION = 256 
TRAINING_EPOCH = 100
PRETRAINED_DIR = "./data/saved_model/model.99-0.88809.h5"
LEARNING_RATE = .0001
BATCH_SIZE = 128

DEMO_DIR = './data/demo_images/'
OUTPUT_DIRS = './data/output/'
FIGURES_DIR = './data/figures/'

FONT_DIR = './data/times-new-roman.ttf'
FONT_SIZE = 30

INFERENCE_BATCH_SIZE = 256
LOSS_WEIGHTS = {'classification':.1,'reconstruction':9}
VAL_SPLIT = .8

LAYER_VISUALIZATION_DIR = './data/2nd_last_layer_visualization/'