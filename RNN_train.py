from dataclasses import dataclass
import os, sys
import numpy as np
from typing import Tuple
import tensorflow as tf
from common_functions import train_stack, predict_stack, adjust_sensitivity, window_sampling, PhysioDatagenerator, stop_training, schedule_learningRate, plot_learnRate_epoch, plot_loss_accuracy
from preprocessingfunctions import SortSPO2HR, SortAccTempEDA, sanity_check_1, necessary_variables, resize_to_uniform_lengths, sanity_check_2_and_DownSamplingAccTempEDA, get_data_dict, plot_varying_recording_time, get_variables
from RNN_model import model
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


tf.keras.backend.clear_session() # clears internal variables so we start all initiations and assignments afresh

@dataclass
class G:
 
  BASE_DIR = './HealthySubjectsBiosignalsDataSet/'
  PATH_TO_SAVED_VARIABLES = './saved_vars.py'

  WHOLE_DICT, CATEGORIES,LABELS_TO_NUMBERS_DICT, NUMBERS_TO_LABELS_DICT = get_variables(PATH_TO_SAVED_VARIABLES)
  #SAVED_CWT_DICT = {i:j/255. for i,j in enumerate(SAVED_CWT_DICT['features'])}
  PERCENT_OF_TRAIN = 1
  
  WINDOW = 60
  OVERLAP = 0.5
  WINDOW_SAMPLING_DICT = {i:j for i,j in enumerate(window_sampling(WHOLE_DICT, window_size = WINDOW, overlap = OVERLAP))}
  SAMPLES_PER_SAMPLE = int(len(WINDOW_SAMPLING_DICT.keys())/len(WHOLE_DICT.keys()))

  TRAIN_FEATURES = train_stack(big_dict = WINDOW_SAMPLING_DICT, train_ratio = PERCENT_OF_TRAIN, sensitivity = SAMPLES_PER_SAMPLE, features = True)
  TRAIN_LABELS = train_stack(WINDOW_SAMPLING_DICT, PERCENT_OF_TRAIN, SAMPLES_PER_SAMPLE, features = False)

  #PREDICT_FEATURES = predict_stack(WINDOW_SAMPLING_DICT, PERCENT_OF_TRAIN, SAMPLES_PER_SAMPLE, features = True)
  #PREDICT_LABELS = predict_stack(WINDOW_SAMPLING_DICT,PERCENT_OF_TRAIN,  SAMPLES_PER_SAMPLE, features = False)

  MODEL_INPUT_SHAPE = TRAIN_FEATURES[0].shape

  TOTAL_TRAIN_DATA = len(TRAIN_FEATURES)
  #TOTAL_VAL_DATA = len(PREDICT_FEATURES)

  EPOCHS = 250

  dp4 = 0.0
  dp1 = 0.3
  dp2 = 0.3
  dp3 = 0.0
  lr = 0.0002

  """
  dp1 = 0.1
  dp2 = 0.5
  dp3 = 0
  lr = 0.0007
  """
  

  TRAIN_BATCH_SIZE = 145#int(TOTAL_TRAIN_DATA / 4) #/8
  #assert(TOTAL_TRAIN_DATA % TRAIN_BATCH_SIZE == 0), "Ensure that the batch size is perfectly divisible by total_train_data"
  
  #VAL_BATCH_SIZE = int(TOTAL_VAL_DATA) # /4
  #assert(TOTAL_VAL_DATA % VAL_BATCH_SIZE == 0), "Ensure teh val_batch_size is perfectly divisible by the total_val_data"

  NUMBER_CLASSES = 4
  TRAIN_STEPS = int(TOTAL_TRAIN_DATA // TRAIN_BATCH_SIZE)
  #VAL_STEPS = int(TOTAL_VAL_DATA // VAL_BATCH_SIZE)


  LOSS = tf.keras.losses.Huber() 
  #LOSS = tf.keras.losses.CategoricalCrossentropy()

  #OPTIMIZER = tf.keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.0)
  OPTIMIZER = tf.keras.optimizers.Adam(learning_rate = lr) #(1e-05 == 0.00001)
  #OPTIMIZER = tf.keras.optimizers.RMSprop(learning_rate = 0.001)#, clipvalue = 0.01)


train_data = PhysioDatagenerator(G.TOTAL_TRAIN_DATA, 
                                  G.TRAIN_FEATURES,
                                  G.LABELS_TO_NUMBERS_DICT,
                                  G.NUMBERS_TO_LABELS_DICT,
                                  batch_size = G.TRAIN_BATCH_SIZE,
                                  shuffle = True,
                                  input_dimention = G.MODEL_INPUT_SHAPE,
                                  augment_data = False,
                                  steps_per_epoch = G.TRAIN_STEPS,
                                  )

train_data_2 = PhysioDatagenerator(G.TOTAL_TRAIN_DATA, 
                                  G.TRAIN_FEATURES,
                                  G.LABELS_TO_NUMBERS_DICT,
                                  G.NUMBERS_TO_LABELS_DICT,
                                  batch_size = G.TRAIN_BATCH_SIZE,
                                  shuffle = False,
                                  input_dimention = G.MODEL_INPUT_SHAPE,
                                  augment_data = False,
                                  steps_per_epoch = G.TRAIN_STEPS,
                                  )

# val_data = PhysioDatagenerator(G.TOTAL_VAL_DATA,
#                                 G.PREDICT_FEATURES,
#                                 G.LABELS_TO_NUMBERS_DICT,
#                                 G.NUMBERS_TO_LABELS_DICT,
#                                 batch_size = G.VAL_BATCH_SIZE,
#                                 shuffle = False,
#                                 input_dimention = G.MODEL_INPUT_SHAPE,
#                                 augment_data = False,
#                                 )


# #
# d = iter(train_data_2)
# samp1 = next(d)
# samp2 = next(d)
# samp1 = next(d)
# samp2 = next(d)
#print(samp2[1])
#print(G.TRAIN_FEATURES.shape)
# samp2 = next(d)
# print((samp2[1]))

#""

#Callbacks = [stop_training(), schedule_learningRate]
Callbacks = [stop_training()]


model = model(G, G.dp1, G.dp2, G.dp3, G.dp4)

print('Traing model...')
history = model.fit(train_data, #batch_size = G.BATCH_SIZE,
                    #steps_per_epoch = G.TRAIN_STEPS,
                    shuffle = True,
                    callbacks = Callbacks,
                    epochs = G.EPOCHS,
                    validation_data = train_data_2,
                    #validation_data = (G.TRAIN_FEATURES, G.TRAIN_LABELS),
                    #validation_data = (G.PREDICT_FEATURES, G.PREDICT_LABELS),
                    #validation_steps = G.TRAIN_STEPS,
                    #validation_batch_size= G.BATCH_SIZE,
                    verbose = 1,
                    )

print('Done!')


import pandas as pd
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.title(f'DropOuts {G.dp4} {G.dp1}, {G.dp2}, {G.dp3}, lr = {G.lr} bs = {G.TRAIN_BATCH_SIZE}, {G.VAL_BATCH_SIZE}')
# plt.savefig('./Plots/Base_model/7.')
plt.show()
print('\n\n')


print('----------Confusion matrix on Training samples-----------------')
features2 = G.TRAIN_FEATURES
labs2 = G.TRAIN_LABELS
predictions = model.predict(features2)
pred_1hot = np.argmax(predictions, axis = 1)
pred_true = np.argmax(labs2, axis = 1)
print(confusion_matrix(pred_true, pred_1hot ))
#print(classification_report(pred_true, pred_1hot))
print('\n\n')

# print('----------Confusion matrix on validation samples-----------------')

# features = G.PREDICT_FEATURES
# labs = G.PREDICT_LABELS
# predictions = model.predict(features)
# pred_1hot = np.argmax(predictions, axis = 1)
# pred_true = np.argmax(labs, axis = 1)
# print(confusion_matrix(pred_true, pred_1hot ))
# #print(classification_report(pred_true, pred_1hot))


#"""


