from dataclasses import dataclass
import os, sys, pickle
import tensorflow as tf
from preprocessingfunctions import get_variables
from physiofunctions import train_stack, predict_stack, adjust_sensitivity, window_sampling, PhysioDatagenerator, stop_training, schedule_learningRate, plot_learnRate_epoch, plot_loss_accuracy
from verifyData import verify
from mymodel import model
from datetime import datetime
from WL_functions import  get_coefficients, WaveletDatagenerator, plot_coefficients, WL_Model_Labels, manual_predict, plot_loss_accuracy#, train_stack, predict_stack, plot_data_from_DATADICT
from WL_Model import wl_model
from sklearn.metrics import confusion_matrix, classification_report


#tf.keras.backend.clear_session() # clears internal variables so we start all initiations and assignments afresh

@dataclass
class G:
  PERCENT_OF_TRAIN = 0.8

  """1
  BASE_DIR = '/content/gdrive/My Drive/PhysioProject1/python-classifier-2020/HealthySubjectsBiosignalsDataSet/'
  PERCENT_OF_TRAIN = 1 # 1 used because all data needs to be processed and stored in variables
  TRAIN_PERCENT = int(PERCENT_OF_TRAIN*len(os.listdir(BASE_DIR)))
  TOTAL_SUBJECT_NUM = len(os.listdir(BASE_DIR)[0:TRAIN_PERCENT])

  SUBJECTS = []
  for i in range(TOTAL_SUBJECT_NUM):
    SUBJECTS.append('Subject'+str(i+1))

  print(f'Subjects {SUBJECTS} used for Training. A total of {len(os.listdir(BASE_DIR))-TOTAL_SUBJECT_NUM} reserved for validation.')

  #SUBJECTS = SUBJECTS[0:10]
  
  SPO2HR, SPO2HR_attributes_dict = SortSPO2HR(BASE_DIR, SUBJECTS)
  AccTempEDA, AccTempEDA_attributes_dict = SortAccTempEDA(BASE_DIR, SUBJECTS)
  sanity_check_1(SPO2HR, AccTempEDA, BASE_DIR, SUBJECTS)

  SPO2HR_target_size, AccTempEDA_target_size, SPO2HR_parameters, AccTempEDA_parameters, common_parameters, Parameters, relax_indices, phy_emo_cog_indices, attributes = necessary_variables('takes_nothing hahaha :)')

  resize_to_uniform_lengths(TOTAL_SUBJECT_NUM, common_parameters, Parameters, SPO2HR_target_size, SPO2HR, AccTempEDA_target_size, AccTempEDA )
  
  AccTempEDA = sanity_check_2_and_MeanOFAccTempEDA(TOTAL_SUBJECT_NUM, common_parameters, Parameters, SPO2HR_target_size, SPO2HR, AccTempEDA_target_size, AccTempEDA, relax_indices, phy_emo_cog_indices)

  INPUT_SHAPE = (7, 299, 300)

  #TRAIN_DATA_PATH = sys.argv[1] 
  NUMBER_CLASSES = 4

  DATA_DICT = get_data_dict(TOTAL_SUBJECT_NUM, common_parameters, Parameters, SPO2HR, AccTempEDA)
  NUMERICAL_LABELS_DICT = {j:i for i,j in enumerate(common_parameters)}
  NUMBERS_TO_LABELS_DICT = {i:j for i,j in enumerate(common_parameters)}
  ALL_COEFFICIENTS = get_coefficients(DATA_DICT)
  WL_ModelLabels = WL_Model_Labels(NUMBER_CLASSES, DATA_DICT)

  #1"""

  
  """2
  NUMBER_CLASSES = 4

  path_2_variables = '/content/gdrive/MyDrive/PhysioProject1/train_vars.py'
  BIG_DATA_DICT, NUMBERS_TO_LABELS_DICT, BIG_ALL_COEFFICIENTS, BIG_WL_Model_Labels, ATTRIBUTES = get_variables(path_2_variables)
  
  TRAIN_COEFFICIENTS = train_stack(BIG_ALL_COEFFICIENTS, PERCENT_OF_TRAIN )
  WL_TRAIN_LABELS = train_stack(BIG_WL_Model_Labels, PERCENT_OF_TRAIN)

  PREDICT_COEFFICIENTS = predict_stack(BIG_ALL_COEFFICIENTS, PERCENT_OF_TRAIN)
  WL_PREDICT_LABELS = predict_stack(BIG_WL_Model_Labels, PERCENT_OF_TRAIN)


  BATCH_SIZE =8
  TOTAL_DATA = TRAIN_COEFFICIENTS.shape[0]
  assert(BATCH_SIZE < 0.5*TOTAL_DATA)
  
  TRAIN_STEPS = int(TOTAL_DATA / BATCH_SIZE)
  EPOCHS = 30

  #LOSS = 'categorical_crossentropy'
  LOSS = tf.keras.losses.Huber()

  OPTIMIZER = tf.keras.optimizers.SGD(learning_rate = 0.001, momentum = 0.0)
  #OPTIMIZER = tf.keras.optimizers.Adam(learning_rate = 0.001) #(1e-05 == 0.00001)
  #OPTIMIZER = tf.keras.optimizers.RMSprop(learning_rate = 0.001)#, clipvalue = 0.01)

 2""" 
  #"""3
  BASE_DIR = '/content/gdrive/MyDrive/PhysioProject1/python-classifier-2020/HealthySubjectsBiosignalsDataSet/'
  PATH_TO_SAVED_VARIABLES = '/content/gdrive/MyDrive/PhysioProject1/python-classifier-2020/saved_vars.py'

  WHOLE_DICT, CATEGORIES,LABELS_TO_NUMBERS_DICT, NUMBERS_TO_LABELS_DICT = get_variables(PATH_TO_SAVED_VARIABLES)
  #SAVED_CWT_DICT = {i:j/255. for i,j in enumerate(SAVED_CWT_DICT['features'])}
  PERCENT_OF_TRAIN = 1

  EPOCHS = 50

  WINDOW = 100 
  assert(WINDOW == 100), 'the saved CWT is compatible with window size of 100'
  OVERLAP = 0.6 
  assert(OVERLAP == 0.6), 'the saved CWT is compatibele with overlap of 60%'

  WINDOW_SAMPLING_DICT = {i:j for i,j in enumerate(window_sampling(WHOLE_DICT, window_size = WINDOW, overlap = OVERLAP))}
  SAMPLES_PER_SAMPLE = int(len(WINDOW_SAMPLING_DICT.keys())/len(WHOLE_DICT.keys()))

  TRAIN_FEATURES = train_stack(WINDOW_SAMPLING_DICT, PERCENT_OF_TRAIN, sensitivity = SAMPLES_PER_SAMPLE, features = True)
  TRAIN_LABELS = train_stack(WINDOW_SAMPLING_DICT, PERCENT_OF_TRAIN, sensitivity = SAMPLES_PER_SAMPLE, features = False)

  #print(TRAIN_LABELS[719:840])

  #PREDICT_FEATURES = predict_stack(WINDOW_SAMPLING_DICT, PERCENT_OF_TRAIN, SAMPLES_PER_SAMPLE, features = True, subject_number = 20)
  #PREDICT_LABELS = predict_stack(WINDOW_SAMPLING_DICT,PERCENT_OF_TRAIN,  SAMPLES_PER_SAMPLE, features = False, subject_number = 20)

  #MODEL_INPUT_SHAPE = (TRAIN_FEATURES[0].shape[0], WINDOW-1, WINDOW) # to be compatible with the images generated by the wavelet transform
  MODEL_INPUT_SHAPE = (3, WINDOW-1, WINDOW) # to be compatible with the images generated by the wavelet transform
  assert(MODEL_INPUT_SHAPE[1]*2 >= 75*2), 'The inception model takes inputs of minumum shape (75,75, channel). Make WINDOW > 75. '

  TOTAL_TRAIN_DATA = len(TRAIN_FEATURES)
  #TOTAL_VAL_DATA = len(PREDICT_FEATURES)

  TRAIN_BATCH_SIZE = int(TOTAL_TRAIN_DATA)
  print(f'Train_batch_size = {TRAIN_BATCH_SIZE}')
  #VAL_BATCH_SIZE = int(TOTAL_VAL_DATA) / 2
  #print(f'Val_batch_size:{VAL_BATCH_SIZE}')
  #assert(TRAIN_BATCH_SIZE < TOTAL_TRAIN_DATA)

  NUMBER_CLASSES = 4
  TRAIN_STEPS = int(TOTAL_TRAIN_DATA // TRAIN_BATCH_SIZE)
  #VAL_STEPS = int(TOTAL_VAL_DATA // VAL_BATCH_SIZE)

  #LOSS = tf.keras.losses.Huber() 
  LOSS = tf.keras.losses.CategoricalCrossentropy()

  lr1 = 0.001 
  
  lr2 = 0.01 #0.001 working pretty well with RMS prop
  OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum = 0.9)
  #OPTIMIZER = tf.keras.optimizers.Adam(lr1) #0.0001 really good
  #Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)
  #OPTIMIZER = tf.keras.optimizers.RMSprop(learning_rate = 0.001)#, clipvalue = 0.01)
  print(f'lr = {lr1} and OPTIMISER: {OPTIMIZER}')
  #3"""

#plot_coefficients(G.ALL_COEFFICIENTS, G.ATTRIBUTES, G, subject_index = 10, specific_subject = False)


#"""4

if __name__ == "__main__":
  #os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
  WL_model = wl_model(G)

  # Set the training parameters
  WL_model.compile(optimizer = G.OPTIMIZER, 
                loss = G.LOSS, 
                metrics = ['accuracy'])

  #WL_model.summary()




  train_data = WaveletDatagenerator(G.TOTAL_TRAIN_DATA, 
                                  G.TRAIN_FEATURES,
                                  G.LABELS_TO_NUMBERS_DICT,
                                  G.NUMBERS_TO_LABELS_DICT,
                                  batch_size = G.TRAIN_BATCH_SIZE,
                                  shuffle = True,
                                  input_dimention = G.MODEL_INPUT_SHAPE,
                                  augment_data = False,
                                  #steps_per_epoch = G.TRAIN_STEPS,
                                  )
                                
  train_data2 = WaveletDatagenerator(G.TOTAL_TRAIN_DATA, 
                                  G.TRAIN_FEATURES,
                                  G.LABELS_TO_NUMBERS_DICT,
                                  G.NUMBERS_TO_LABELS_DICT,
                                  batch_size = int(G.TRAIN_BATCH_SIZE/3),
                                  shuffle = False,
                                  input_dimention = G.MODEL_INPUT_SHAPE,
                                  augment_data = False,
                                  #steps_per_epoch = G.TRAIN_STEPS,
                                  )


  # val_data = WaveletDatagenerator(G.TOTAL_VAL_DATA,
  #                               G.PREDICT_FEATURES,
  #                               G.LABELS_TO_NUMBERS_DICT,
  #                               G.NUMBERS_TO_LABELS_DICT,
  #                               batch_size = G.VAL_BATCH_SIZE,
  #                               shuffle = False,
  #                               input_dimention = G.MODEL_INPUT_SHAPE,
  #                               augment_data = False,
  #                               )


  # import numpy as np
  # import pywt
  # d = iter(train_data)
  # samp1 = next(d)
  # samp2 = next(d)
  # samp3 = next(d)
  # samp4 = next(d)

  # WL_data = np.vstack((samp1[0], samp2[0], samp3[0], samp4[0]))
  # WL_data_labels = np.vstack((samp1[1], samp2[1], samp3[1], samp4[1]))

  # WL_DICT = {'features': WL_data, 'Labels': WL_data_labels}
 

  # print(type(samp1[0]))
  # print(samp1[0][0].shape)
  # print(samp1[1].shape)
  # samp1 = next(d)
  # samp1 = next(d)
  # samp1 = next(d)
  # samp1 = next(d)
  # samp1 = next(d)
  # print('Nochange')
  #print(samp1[0].shape) # (30, 7, 60)
  # # #temp = np.transpose(samp1[0][0])
  # scales = range(1,60)

  # import matplotlib.pyplot as plt
  # fig, axs = plt.subplots(1,7, figsize=(28, 4), facecolor='w', edgecolor='k')
  # fig.subplots_adjust(hspace = .5, wspace=.001)
  # axs = axs.ravel()
  # for j in range(7):
  #   axs[j].matshow(samp1[0][0][j,:,:])
  # plt.show()

#4"""

  history = WL_model.fit(train_data, #batch_size = G.BATCH_SIZE,
                    #steps_per_epoch = G.TRAIN_STEPS,
                    shuffle = True,
                    #callbacks = Callbacks,
                    epochs = G.EPOCHS,
                    validation_data = train_data2,
                    #validation_data = (G.TRAIN_FEATURES, G.TRAIN_LABELS),
                    #validation_data = (G.PREDICT_FEATURES, G.PREDICT_LABELS),
                    #validation_steps = G.TRAIN_STEPS,
                    #validation_batch_size= G.BATCH_SIZE, 
  )
                 

#   manual_predict(WL_model, G.ALL_COEFFICIENTS)

  # plot_loss_accuracy(history)

#"""5
  import pandas as pd
  import matplotlib.pyplot as plt
  pd.DataFrame(history.history).plot(figsize=(8,5))
  plt.title(f'lr1 = {G.lr1}, OPT:{G.OPTIMIZER}')
  plt.show()


  # import numpy as np
  # import keras

  # print('----------Confusion matrix for Training-----------------')
  # labs2 = G.TRAIN_LABELS
  # predictions = WL_model.predict(G.SAVED_CWT_TRAIN_STACK)
  # pred_1hot = np.argmax(predictions, axis = 1)
  # pred_true = np.argmax(labs2, axis = 1)
  # print(confusion_matrix(pred_true, pred_1hot ))
  # print(classification_report(pred_true, pred_1hot))
  # print('\n\n\n\n\n\n\n')

  # print('----------Confusion matrix for validation-----------------')

  # features = G.SAVED_CWT_PREDICT_STACK
  # labs = G.PREDICT_LABELS
  # predictions = WL_model.predict(features)
  # pred_1hot = np.argmax(predictions, axis = 1)
  # pred_true = np.argmax(labs, axis = 1)
  # print(confusion_matrix(pred_true, pred_1hot ))
  # # print(classification_report(pred_true, pred_1hot))


#5"""

"""6
  # FOR DEBUGGING THE DATA_DICT WHEN APPLYING WAVELET TRANSFROMS
  # Basically this writes the needed variables into the file path so they can be accessed
  # on the colab workspace.


G.path_2_variables = '/content/gdrive/My Drive/PhysioProject1/train_vars.py' # path of file for needed ariables

try:
  os.remove(G.path_2_variables)
  with open(G.path_2_variables, 'wb') as File:
    pickle.dump(G.DATA_DICT, File)
    pickle.dump(G.NUMBERS_TO_LABELS_DICT, File)
    pickle.dump(G.ALL_COEFFICIENTS, File)
    pickle.dump(G.WL_ModelLabels, File)
    pickle.dump(G.attributes, File)
    pickle.dump(G.temp_DICT, File)
    
except FileNotFoundError:
  with open(G.path_2_variables, 'wb') as File:
    pickle.dump(G.DATA_DICT, File)
    pickle.dump(G.NUMBERS_TO_LABELS_DICT, File)
    pickle.dump(G.ALL_COEFFICIENTS, File)
    pickle.dump(G.WL_ModelLabels, File)
    pickle.dump(G.attributes, File)
    pickle.dump(G.temp_DICT, File)
#6"""
