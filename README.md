# Handwritten-Digit-To-Digital-Digit

##### This model maps handwritten digit(mnist) to fine 'DIGITAL' digit. We create sythetic IMAGE label for each input image with time-new-roman.ttf font using PILLOW libary. Our model is a multi-task learner with main objective being to map input digit image to fine-grained image while classifiying the input acts as secondary objective.

  To make it work with your workstation, follow the following steps:

    1. Clone the repo.
    2. Setup your enviroment with requirement.txt file.

Basic Info:
1. sythetic_label.py was used to generate synthetic images
2. data_preparation.py generates train, test image. Models keep part of it training data for validation 
3. For tensorboard visualization run "tensorboard --logdir="./data/logs" from base directory 
4. To view model architecture, have a look on ./data/model_architecture.png 

  With config.py,
  You can change the outcome of the main.py file as your liking.
  Change the following CONSTANT to change:

    1. TRAINING: For training
    2. TESTING : For Testing
    3. DEMO : Demo with image in DEMO_DIR
    4. EVALUATE_PERFORMANCE : To Evaluate Performance and generate evaluation metrices & Plot
    5. VISULIZE_2ND_LAST_LAYER : To visualize 2nd last layer Activation
    6. POSTPROCESS : To Remove noise(But image has staircase effect)
    7. PRINT_SUMMARY : To print model summary-
    8. TESTING_VIS: To visualize testing-  [Default: 256; this number of arbitarily selected images will be saved in the OUTPUT_DIRS during testing]
                                            [Change it to 0 to turn it off]
    9. You can also save concatenated image of input & output image in utils.py function

  Other parameters you can change:

    Training epoch : TRAINING_EPOCH = 100
    Pretrainded Directory : PRETRAINED_DIR = './data/saved_model/model.h5' [for Testing or Retrianing]
    Learing Rate : LEARNING_RATE = .0001  [Learning rate of Nadam Optimizer]
    Batch size : BATCH_SIZE = 128 [11GB VRAM]
    Put your image here to visualize : DEMO_DIR = './data/demo_images/'
    Demo output directory during testing : OUTPUT_DIRS = './data/output'

    FONT_DIR = './data/times-new-roman.ttf'  [Font of Digital Output]
    FONT_SIZE = 30 [Font size of Digital Output]

    INFERENCE_BATCH_SIZE = 256 [Double of Training Batch Size]

    LOSS_WEIGHTS = {'classification':.1,'reconstruction':9} [Weight of Classification & Reconstruction Loss]
    VAL_SPLIT = .8 [ 20% of data will be kept for validation]
    LAYER_VISUALIZATION_DIR = 'second_last_layer_vis' [Layer Visualization Output]