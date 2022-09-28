import numpy as np, os, sys, joblib, scipy, Cython, shutil, pickle, time, copy, keras
from dataclasses import dataclass
#!pip install scikit-learn
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd




def adjust_sensitivity(Dict, sensitivity):
  temp_list = []
  for i in range(len(Dict.keys())):
    temp1 = np.array(np.hsplit(Dict[i], sensitivity))
    for j in range(sensitivity):
      temp_list.append(temp1[j])
  return temp_list




def train_stack(big_dict, train_ratio, sensitivity = 1, features = True ):
  Relax_index, PhysicalStress_index, EmotionalStress_index, CognitiveStress_index = 80, 100, 120, 140
  relax, physical, emotional, cognitive = [], [], [], []
  if features:
    if sensitivity == 1:
      #stop = int(subject_number*train_ratio)
      relax_indices = [i for i in range(0, Relax_index)]
      for i in relax_indices:
        relax.append(big_dict[i])

      #physical = big_dict[20+stop :40]
      physical_indices = [i for i in range(PhysicalStress_index - 20, PhysicalStress_index)]
      for i in physical_indices:
        physical.append(big_dict[i])

      #emotional = big_dict[40+stop :60]
      emotional_indices = [i for i in range(EmotionalStress_index-20, EmotionalStress_index)]
      for i in emotional_indices:
        emotional.append(big_dict[i])

      #cognitive = big_dict[60+stop :80]
      cognitive_indices = [i for i in range(CognitiveStress_index-20, CognitiveStress_index)]
      for i in cognitive_indices:
        cognitive.append(big_dict[i])

    else:
      #start = 140*sensitivity
      #stop = int(subject_number*train_ratio)*sensitivity
      #relax_indices = [i for i in range(0, stop)]
      for i in range(0, Relax_index*sensitivity):
        relax.append(big_dict[i])

      #physical = big_dict[20+stop :40]
      physical_indices = [i for i in range(sensitivity*(PhysicalStress_index-20), PhysicalStress_index*sensitivity)]
      for i in physical_indices:
        physical.append(big_dict[i])

      #emotional = big_dict[40+stop :60]
      emotional_indices = [i for i in range(sensitivity*(EmotionalStress_index-20), EmotionalStress_index*sensitivity)]
      for i in emotional_indices:
        emotional.append(big_dict[i])

      #cognitive = big_dict[60+stop :80]
      cognitive_indices = [i for i in range(sensitivity*(CognitiveStress_index-20), CognitiveStress_index*sensitivity)]
      for i in cognitive_indices:
        cognitive.append(big_dict[i])

  elif features == False:
    if sensitivity == 1:
      #stop = int(subject_number*train_ratio)

      relax_indices = [i for i in range(0, Relax_index)]
      for i in relax_indices:
        relax.append(np.eye(4)[0])

      #physical = big_dict[20+stop :40]
      physical_indices = [i for i in range(PhysicalStress_index - 20, PhysicalStress_index)]
      for i in physical_indices:
        physical.append(np.eye(4)[1])

      #emotional = big_dict[40+stop :60]
      emotional_indices = [i for i in range(EmotionalStress_index - 20, EmotionalStress_index)]
      for i in emotional_indices:
        emotional.append(np.eye(4)[2])

      #cognitive = big_dict[60+stop :80]
      cognitive_indices = [i for i in range(CognitiveStress_index - 20, CognitiveStress_index)]
      for i in cognitive_indices:
        cognitive.append(np.eye(4)[3])

    else:
      #start = subject_number*sensitivity
      #stop = int(subject_number*train_ratio)*sensitivity
      relax_indices = [i for i in range(0, Relax_index*sensitivity)]
      for i in relax_indices:
        relax.append(np.eye(4)[0])

      #physical = big_dict[20+stop :40]
      physical_indices = [i for i in range(sensitivity*(PhysicalStress_index-20), PhysicalStress_index*sensitivity)]
      for i in physical_indices:
        physical.append(np.eye(4)[1])

      #emotional = big_dict[40+stop :60]
      emotional_indices = [i for i in range(sensitivity*(EmotionalStress_index-20), EmotionalStress_index*sensitivity)]
      for i in emotional_indices:
        emotional.append(np.eye(4)[2])

      #cognitive = big_dict[60+stop :80]
      cognitive_indices = [i for i in range(sensitivity*(CognitiveStress_index-20), CognitiveStress_index*sensitivity)]
      for i in cognitive_indices:
        cognitive.append(np.eye(4)[3])


  return np.vstack((np.array(relax), np.array(physical), np.array(emotional), np.array(cognitive)))




# PREDICTION
def predict_stack(big_dict, train_ratio, sensitivity = 1, features = True, subject_number = 20):
  relax, physical, emotional, cognitive = [], [], [], []
  
  if features:
    if sensitivity == 1:
      stop = int(subject_number*train_ratio)
      #relax_indices = [i for i in range(stop, 20)]
      for i in range(stop, subject_number):
        relax.append(big_dict[i])

      #physical = big_dict[20+stop :40]
      #physical_indices = [i for i in range(20+stop, 40)]
      for i in range(subject_number+stop, subject_number*2):
        physical.append(big_dict[i])

      #emotional = big_dict[40+stop :60]
      #emotional_indices = [i for i in range(40+stop, 60)]
      for i in range(subject_number*2+stop, subject_number*3):
        emotional.append(big_dict[i])

      #cognitive = big_dict[60+stop :80]
      #cognitive_indices = [i for i in range(60+stop, 80)]
      for i in range(subject_number*3+stop, subject_number*4):
        cognitive.append(big_dict[i])


    else:
      stop = int(subject_number*train_ratio)*sensitivity
      block = subject_number*sensitivity

      #relax_indices = [i for i in range(stop, block)]
      for i in range(stop, block):
        relax.append(big_dict[i])

      #physical = big_dict[20+stop :40]
      #physical_indices = [i for i in range(block+stop, block*2)]
      for i in range(block+stop, block*2):
        physical.append(big_dict[i])

      #emotional = big_dict[40+stop :60]
      #emotional_indices = [i for i in range(block*2+stop, block*3)]
      for i in  range(block*2+stop, block*3):
        emotional.append(big_dict[i])

      #cognitive = big_dict[60+stop :80]
      #cognitive_indices = [i for i in range(block*3+stop, block*4)]
      for i in range(block*3+stop, block*4):
        cognitive.append(big_dict[i])


  elif features == False:
    if sensitivity == 1:
      stop = int(subject_number*train_ratio)
      block = subject_number
      for i in range(stop, block):
        relax.append(np.eye(4)[0])

      for i in range(block+stop, block*2):
        physical.append(np.eye(4)[1])

      for i in range(block*2+stop, block*3):
        emotional.append(np.eye(4)[2])

      for i in range(block*3+stop, block*4):
        cognitive.append(np.eye(4)[3])

    else:
      stop = int(subject_number*train_ratio)*sensitivity
      block = subject_number*sensitivity

      #relax_indices = [i for i in range(stop, block)]
      for i in range(stop, block):
        relax.append(np.eye(4)[0])

      #physical = big_dict[20+stop :40]
      #physical_indices = [i for i in range(block+stop, block*2)]
      for i in range(block+stop, block*2):
        physical.append(np.eye(4)[1])

      #emotional = big_dict[40+stop :60]
      #emotional_indices = [i for i in range(block*2+stop, block*3)]
      for i in range(block*2+stop, block*3):
        emotional.append(np.eye(4)[2])

      #cognitive = big_dict[60+stop :80]
      #cognitive_indices = [i for i in range(block*3+stop, block*4)]
      for i in range(block*3+stop, block*4):
        cognitive.append(np.eye(4)[3])     


  return np.vstack((np.array(relax), np.array(physical), np.array(emotional), np.array(cognitive)))






def window_sampling(samples_dict, window_size = 100, overlap = 0.6):

  array_width = samples_dict[0].shape[1] #stores the allowable range of indices
  percent_overlap = 1 - overlap #this is used instead to give generate the expected overlap. Using just overlap generates (1-overlap) overlap
  stride = int(percent_overlap*window_size)
  assert(stride > 0 and stride != 1), "Stride too small. Reduce the value of overlap."
  max_samples = int(((array_width - window_size)/stride) + 1)
  assert(window_size<array_width), "Window size should be less than total array width"
  assert(overlap != 1), "Percentage of overlap should be less than 100% or 1"

  temp_samples = [] #to keep generated samples

  temp_dict = {} #keeps generated indices for debugging
  #keeps a zipped list of generated indices for dubugging----somehow unnecessary
  #zipped_indices = list( zip((int(percent_overlap*window_size*i) for i in range(max_samples)), (int(window_size+percent_overlap*window_size*i) for i in range(max_samples))))
  for j in range(len(samples_dict.keys())):
    for i in range(max_samples):
      start = int(percent_overlap*window_size*i)
      stop = int(window_size+(start))
      #temp_dict[i] = [start, stop]

      assert(stop <= array_width),f'Allowabe max_index = {array_width}---Last generated index = {stop}.'
      temp = samples_dict[j][:, start : stop]
      temp_samples.append(temp)
  print(f'Original subject number = {len(samples_dict.keys())} \nSamples per sample = {max_samples} \nTotal generated = {len(temp_samples)}')
  return temp_samples #,temp_dict#, zipped_indices



# CALLBACK FUNCTIONS
class stop_training(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if (logs.get('accuracy') > 0.99) and (logs.get('val_accuracy') > 0.99):
      print("\nReached 94.0% accuracy and over 90% val accuracy -> so cancelling training!")
      self.model.stop_training = True

schedule_learningRate = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch/20))


graphdir="/content/gdrive/MyDrive/PhysioProject1/python-classifier-2020/RNN_graph"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=graphdir)



import matplotlib.pyplot as plt

class PhysioDatagenerator(tf.keras.utils.Sequence):
#class PhysioDatagenerator(tf.keras.utils.Sequence):
  number_of_generated_samples = 0
  def __init__(self, total_subject_num,
                    data_dict,
                    labels_to_numbers_dict,
                    numbers_to_labels_dict,
                    input_dimention = (7, 300),
                    batch_size = 10,
                    num_classes = 4,
                    num_channels = 1,
                    shuffle = False,
                    augment_data = False,
                    steps_per_epoch = 0.5,
                    ):

    self.total_subject_num = total_subject_num
    self.data_dict = {i:j for i,j in enumerate(data_dict)} # this converts the numpy array into a dictionary with their indices as keys
    self.labels_to_numbers_dict = labels_to_numbers_dict
    self.numbers_to_labels_dict = numbers_to_labels_dict
    self.input_dimention = input_dimention
    self.batch_size = batch_size
    self.augment_data = augment_data
    self.steps_per_epoch = steps_per_epoch

    # if assertion thrown, batch_size = 10 is the default size. Check the Train file and pass the right size to the datagenerator
    #assert(self.batch_size < 0.5*self.total_subject_num)
    self.num_classes = num_classes
    self.num_channels = num_channels
    self.shuffle = shuffle
    self.on_epoch_end()
    self.seed = 0
    if self.shuffle:
      print(f'Total train samples = {self.total_subject_num}')
    else:
      print(f'Total validation samples = {self.total_subject_num}')


    # if overlap_sampling:
    #   self.array_width = 300 #input_dimention[1]
    #   self.percent_overlap = 1-percent_overlap
    #   self.stride = int(percent_overlap*window_size)
    #   assert(self.stride > 0 and self.stride != 1), "Stride too small. Reduce the value of overlap."
    #   self.max_samples = int(((self.array_width - window_size)/self.stride) + 1)
    #   assert(window_size <= self.array_width), f"Window size should be less than total array width. Window_size:{window_size}, array_width:{self.array_width}. Check and make sensitivity=1 so that array has original dimention"
    #   assert(percent_overlap != 1), "Percentage of overlap should be less than 100% or 1"
    #   self.zipped_indices = list( zip((int(percent_overlap*window_size*i) for i in range(self.max_samples)), (int(window_size+percent_overlap*window_size*i) for i in range(self.max_samples))))


    """
    # THIS IS USED FOR VERIFYING THE DATA FROM THE GENERATOR MATCHES
    # THAT OF THE RECORDED
    # self.subject1spo2 = np.hstack([self.data_dict[0], self.data_dict[1], self.data_dict[2], self.data_dict[3]])[0,:]
    # import pickle, os
    # os.rmdir('/content/gdrive/MyDrive/PhysioProject1/python-classifier-2020/vars.py')
    # os.mkdir('/content/gdrive/MyDrive/PhysioProject1/python-classifier-2020/vars.py') 
    # with open('/content/gdrive/MyDrive/PhysioProject1/python-classifier-2020/vars.py', 'wb') as f:
    # # b is added cos the data we are writing may be binary .... in the case of ndarrays
    #   pickle.dump(self.subject1spo2, f)
    """

  def __getitem__(self, index):

    batch_to_load = self.indexes[index*self.batch_size: (index+1)*self.batch_size]
    #print(f'Batch to load = {batch_to_load}')    
    X, y = self.__data_generation(batch_to_load)
    
    # wait = True
    # while wait:
    #   for i in y:
    #     print(self.numbers_to_labels_dict[np.where(i==1)[0][0]])
    #   wait = False
    #print(f'{(y[0])}')
    #if wait == False:

    return X, y


  def __len__(self):
    return int((len(self.data_dict.keys()))/self.batch_size)


  def on_epoch_end(self):
    """
    Shuffles the list of indices for random generation
    """
    self.indexes = np.arange(len(self.data_dict.keys()))
    self.seed = np.random.randint(0,10)
    if self.shuffle == True:
      np.random.shuffle(self.indexes)
    


  def __data_generation(self, batch_indices):
    """
    The batch indices list contains the indices of the next batch to fetch. [0,1,2,3] fetches the first, sec, third and
    fourth sampples if our batch size is 4
    """
    # this initialises empty values and label array to be filled later
    X = np.empty((self.batch_size, *self.input_dimention))
    y = np.empty((self.batch_size), dtype=int)
    PhysioDatagenerator.number_of_generated_samples += 1

    
    #print(self.data_dict.keys())

    # if self.augment_data:
    #   print('Augmenting data...')
    # else:
    #   print('No Data Augmentation')

    for i,j in enumerate(batch_indices):
      # print(PhysioDatagenerator.number_of_generated_samples)
      # print(self.steps_per_epoch)
      if self.augment_data and (PhysioDatagenerator.number_of_generated_samples % 4) == 0:
        #print(f'sample_number = {PhysioDatagenerator.number_of_generated_samples}')
        np.random.seed(self.seed)
        np.random.shuffle(self.data_dict[j])
        X[i,] = self.data_dict[j]

      else:
        #print('Non augmented')
        #temp_sample = np.random.randint(self.max_samples)
        #X[i,] = self.data_dict[j][:, self.zipped_indices[temp_sample][0]: self.zipped_indices[temp_sample][1] ] # original 7-channel data
        X[i,] = self.data_dict[j]

      first_quartile = 0.25*self.total_subject_num # first quarter of data are 'Relax', indices [0-19], if total number of samples is say 80
      second_quartile = 0.5*self.total_subject_num # 'second quarter' of data are 'PhysicalStress', indices [20-39]
      third_quartile = 0.75*self.total_subject_num # 'third quarter' of data are 'EmotionalStress', indices [40-59]

      if j < first_quartile:
        y[i] = self.labels_to_numbers_dict['Relax']
      elif first_quartile <= j < second_quartile:
        y[i] = self.labels_to_numbers_dict['PhysicalStress']
      elif second_quartile <= j < third_quartile:
        y[i] = self.labels_to_numbers_dict['EmotionalStress']
      else:
        y[i] = self.labels_to_numbers_dict['CognitiveStress']

      """
      remember data_dict is organised this way
      data_dict = {0:[], 1:[], 2:[], -----, 79:[]}
      first 20 samples are relaxex
      next 20 physical
      next 20 emotional
      last 20 cognitive
      We know this from the cration of the dictionary in get_data_dict() function where we stacked them in a specific order
      """
    return X, keras.utils.np_utils.to_categorical(y, num_classes=self.num_classes)






def plot_learnRate_epoch(epoch_number, history):

  base = '/content/gdrive/MyDrive/PhysioProject1/python-classifier-2020/Plots'
  target = os.path.join(base, 'learningRate.png')

  try:
    os.remove(target)
    with open(target, 'wb') as File:
      lrs = 1e-8 * (10 ** (np.arange(epoch_number) / 20))
      plt.figure(figsize=(10, 6))
      plt.grid(True)
      plt.semilogx(lrs, history.history["loss"])
      plt.tick_params('both', length=10, width=1, which='both')
      #plt.axis([1e-8, 1e-3, 0, 30])
      plt.title("Learning Rate Schedule")
      plt.xlabel('Learning Rate')
      plt.ylabel('Loss')
      plt.savefig(target)
      plt.clf()

  except FileNotFoundError:
    with open(target, 'wb') as File:
      lrs = 1e-8 * (10 ** (np.arange(epoch_number) / 20))
      plt.figure(figsize=(6, 3))
      plt.grid(True)
      plt.semilogx(lrs, history.history["loss"])
      plt.tick_params('both', length=10, width=1, which='both')
      plt.axis([1e-8, 1e-3, 0, 30])
      plt.title("Learning Rate")
      plt.xlabel('Learning Rate')
      plt.ylabel('Epoch Number')
      plt.savefig(target)
      plt.clf()


def plot_loss_accuracy(history,):

  base = '/content/gdrive/MyDrive/PhysioProject1/python-classifier-2020/Plots'
  target = os.path.join(base, 'Accuracy_Loss.png')

  try:
    os.remove(target)
    with open (target, 'wb') as File:
      accuracy = history.history['accuracy']
      loss = history.history['loss']
      epochs=range(len(loss)) 

      plt.plot(epochs, accuracy, epochs, loss)
      plt.title='Accuracy and Loss'
      xlabel='Epochs'
      plt.legend(['Accuracy', 'Loss'])
      plt.savefig(target)
      plt.clf()

      # # Only plot the last 80% of the epochs
      # zoom_split = int(epochs[-1] * 0.2)
      # epochs_zoom = epochs[zoom_split:]
      # accuracy_zoom = accuracy[zoom_split:]
      # loss_zoom = loss[zoom_split:]

      # plt.plot(epochs_zoom, accuracy_zoom, epochs_zoom, loss_zoom)
      # title='Zoomed Accuracy and Loss'
      # xlabel='Epochs'
      # plt.legend('Accuracy', 'Loss')
      # plt.savefig(target)

  except FileNotFoundError:
    with open(target, 'wb') as File:
      accuracy = history.history['accuracy']
      loss = history.history['loss']
      epochs=range(len(loss)) 

      plt.plot(epochs, accuracy, epochs, loss)
      plt.title='Accuracy and Loss'
      xlabel='Epochs'
      plt.legend(['Accuracy', 'Loss'])
      plt.savefig(target)
      plt.clf()

      # # Only plot the last 80% of the epochs
      # zoom_split = int(epochs[-1] * 0.2)
      # epochs_zoom = epochs[zoom_split:]
      # accuracy_zoom = accuracy[zoom_split:]
      # loss_zoom = loss[zoom_split:]

      # plt.plot(epochs_zoom, accuracy_zoom, epochs_zoom, loss_zoom)
      # title='Zoomed Accuracy and Loss'
      # xlabel='Epochs'
      # plt.legend('Accuracy', 'Loss')
      # plt.savefig(target)



  

