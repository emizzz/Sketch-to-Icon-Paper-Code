#*********************************************************************************************************************************************
from PIL import Image, ImageOps  
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from google.colab.patches import cv2_imshow
import cv2
import random
import tensorflow as tf
import math

#*********************************************************************************************************************************************
def load_data(path=None, data=None, size=None, _3d=None, invert=False, randomize=False, rand_seed=42):
    x_train, y_train = None, None
    x_out_test, x_out_train = [], []
    assert (path or data) and not (path and data), "You must provide at least the path or the data, not both"

    try:
        (x_train, y_train) = pickle.load(open(path, "rb")) if path else data
    except:
        x_train = pickle.load(open(path, "rb")) if path else data

    
    # ----------------data----------------------
    for i, x in enumerate(x_train):
        if len(x.shape) == 1:
          x = x.reshape((int(math.sqrt(x.shape[0])), int(math.sqrt(x.shape[0]))))
          
        img = Image.fromarray(np.uint8(x))
        if size and img.size[0] != size and img.size[1] != size:
            img = img.resize((size, size), Image.NEAREST)
        if invert:
            img = ImageOps.invert(img)
        if _3d:
            img = img.convert('RGB')
            img = np.asarray(img)
        else:
            img = img.convert('L')
            img = np.asarray(img)
            img = img[..., np.newaxis]
  
        x_out_train.append(img)

    if randomize:
        idx_train = np.random.RandomState(seed=rand_seed).permutation(len(x_out_train))
        x_out_train  = x_out_train[idx_train]
        if np.any(y_train != None):
          y_out_train = y_out_train[idx_train]


    x_out_train = np.asarray(x_out_train)
    if np.any(y_train != None):
      y_train = np.asarray(y_train)

    if np.any(y_train != None):
      return ((x_out_train, y_train))
    else:
      return ((x_out_train))

#*********************************************************************************************************************************************



#*********************************************************************************************************************************************
def split_dataset(_X, _y_string=np.array([]), _validation_size=0, _test_size=0, _random_seed=None, stratify=True):
  x_train, x_valid, x_test = np.array([]), np.array([]), np.array([])
  y_string_train, y_string_valid, y_string_test = np.array([]), np.array([]), np.array([])
  
  x_train, y_string_train = _X, _y_string

  
  if _test_size != 0:
    x_train, x_test, y_string_train, y_string_test = train_test_split(
        x_train, y_string_train, 
        test_size=_test_size, 
        random_state=_random_seed if _random_seed else None, 
        shuffle=True if _random_seed else False,
        stratify=False if not stratify else y_string_train
    )
  
  if _validation_size != 0:
    x_train, x_valid, y_string_train, y_string_valid = train_test_split(
        x_train, y_string_train, 
        test_size=_validation_size, 
        random_state=_random_seed if _random_seed else None, 
        shuffle=True if _random_seed else False,
        stratify=False if not stratify else y_string_train
    )

  
  print("created: x_train: ", x_train.shape)
  print("created: x_valid: ", x_valid.shape)
  print("created: x_test: ", x_test.shape)
  print("created: y_string_train: ", y_string_train.shape)
  print("created: y_string_valid: ", y_string_valid.shape)
  print("created: y_string_test: ", y_string_test.shape)
  print("created: num classes: ", len(np.unique(y_string_train)), len(np.unique(y_string_valid)), len(np.unique(y_string_test)))

  return x_train, x_valid, x_test, y_string_train, y_string_valid, y_string_test
#*********************************************************************************************************************************************



#*********************************************************************************************************************************************
def data_preprocessing(x):
  return (x.astype('float32')) / 255.
#*********************************************************************************************************************************************



#*********************************************************************************************************************************************
def labels_preprocessing(y_train, y_valid=[], y_test=[]):
    all_labels = list(set(y_train.tolist() + y_valid.tolist() + y_test.tolist() ))  
    all_labels.sort()
    all_labels = {l: i for i, l in enumerate(all_labels)}

    y_train = np.array([all_labels[l] for l in y_train])
    y_valid = np.array([all_labels[l] for l in y_valid])
    y_test = np.array([all_labels[l] for l in y_test])

    return y_train, y_valid, y_test

#*********************************************************************************************************************************************



#*********************************************************************************************************************************************
'''
checks if the 2 datasets match (have the same classes), otherwise this function 
try to crops the dataset with more classes
'''
def check_dataset_classes(x_1, x_2, y_1, y_2):
  
  if len(set(y_1)) != len(set(y_2)):
    print("datasets don't match")

  d1_d2, d2_d1 = set(y_1) - set(y_2), set(y_2) - set(y_1)

  for class_to_delete in d1_d2:
    to_delete_indexes = np.where(y_1 == class_to_delete)
    x_1 = np.delete(x_1, to_delete_indexes, axis=0)
    y_1 = np.delete(y_1, to_delete_indexes)

  for class_to_delete in d2_d1:
    to_delete_indexes = np.where(y_2 == class_to_delete)
    x_2 = np.delete(x_2, to_delete_indexes, axis=0)
    y_2 = np.delete(y_2, to_delete_indexes)

  print("dataset matched")
  return x_1, x_2, y_1, y_2
#*********************************************************************************************************************************************



#*********************************************************************************************************************************************
'''
  colab path to cv2 "show_img"
'''

def show_img(img):
  assert len(img.shape) == 2 or len(img.shape) == 3
  if len(img.shape) > 2:
    img = img[:, :, 0]

  if img.max() <= 1:
    img = img*255

  cv2_imshow(img)
#*********************************************************************************************************************************************



#*********************************************************************************************************************************************
'''
  inverse of tf.keras.utils.to_categorical method
'''
def from_categorical(data):
  return np.argmax(data, axis=-1)
#*********************************************************************************************************************************************


#*********************************************************************************************************************************************
'''
checks if the 2 datasets match (have the same classes), otherwise this function 
try to crops the dataset with more classes
'''
def check_dataset_classes(x_1, x_2, y_1, y_2):
  
  if len(set(y_1)) != len(set(y_2)):
    print("datasets don't match")

  d1_d2, d2_d1 = set(y_1) - set(y_2), set(y_2) - set(y_1)

  for class_to_delete in d1_d2:
    to_delete_indexes = np.where(y_1 == class_to_delete)
    x_1 = np.delete(x_1, to_delete_indexes, axis=0)
    y_1 = np.delete(y_1, to_delete_indexes)

  for class_to_delete in d2_d1:
    to_delete_indexes = np.where(y_2 == class_to_delete)
    x_2 = np.delete(x_2, to_delete_indexes, axis=0)
    y_2 = np.delete(y_2, to_delete_indexes)

  print("dataset matched")
  return x_1, x_2, y_1, y_2
#*********************************************************************************************************************************************


#*********************************************************************************************************************************************
def shuffle_with_same_indexes(arr1, arr2, seed=None):
  assert len(arr1) == len(arr2), "Vectors must have the same length"
  if seed == None:
    print("random_seed not provided") 
    
  rand_idxs = np.random.RandomState(seed=seed).permutation(len(arr1))
  
  if isinstance(arr1, list) and isinstance(arr2, list):
    return [arr1[i] for i in rand_idxs], [arr2[i] for i in rand_idxs]

  elif isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray):
    return arr1[rand_idxs], arr2[rand_idxs]

  else:
    print("different data types")
    return None
#*********************************************************************************************************************************************




#*********************************************************************************************************************************************
'''
example of use:
cutom_aug = CustomAug()
datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=cutom_aug.custom_preprocessing)
train_gen = datagen.flow(x_train_icons, y_train_icons, batch_size=batch_size)

history = net.fit(
  train_gen,
  validation_data=(x_valid_icons, y_valid_icons),
  steps_per_epoch=len(x_train_icons) // batch_size, epochs=n_epoch
)
'''
class CustomAug:
    def __init__(self, _custom_tranformations, _default_tranformations={}):
      self.augmentator = tf.keras.preprocessing.image.ImageDataGenerator()
    
      custom_tranformations = {
        'rescale' : False,   
        'pad' : False,                           
        'horizontal_flip' : False,              # (h-flip) it doesn't work in tf.keras
        'erosion' : False,                       
        'noise' : False, 
        'half_aug' : False,
      }
      self.custom_tranformations =  {**custom_tranformations, **_custom_tranformations}

      default_tranformations = {
        'theta': 10,
        'tx' : 5,
        'ty' : 5,
        'zx' : 1,
        'zy' : 1,
      }
      self.default_tranformations =  {**default_tranformations, **_default_tranformations}

    def custom_preprocessing(self, im):
      rand_zoom = random.uniform(1, 2)

      self.default_transformations = {   
        'theta': random.uniform(-10, 10),
        'tx' : random.uniform(-5, 5),
        'ty' : random.uniform(-5, 5),
        'zx' : rand_zoom,
        'zy' : rand_zoom,
      }
      
      if self.custom_tranformations['half_aug']:
        if np.random.rand() > .5: return self.custom_global_trans(im)
      
      im = self.custom_pre_trans(im)
      
      im =  self.augmentator.apply_transform(im, self.default_transformations)
      
      return self.custom_global_trans(im)
      

    def custom_pre_trans(self, img):
      
      # the tf.keras flip doesn't seem to work
      if self.custom_tranformations['horizontal_flip']:
        if np.random.rand() > .5: 
          img = self.horizontal_flip(img)

      #erosion
      if self.custom_tranformations['erosion']:
        img = self.erode(img)

      # pad the image
      if self.custom_tranformations['pad']:
        img = self.pad_img(img)
      
      return img

    def custom_global_trans(self, img):

      if self.custom_tranformations['rescale']:
        img = data_preprocessing(img)

      return img

    @staticmethod
    def horizontal_flip(img):
      return img[:, ::-1]

    def pad_img(self, img):
      w, h, d = img.shape
      
      new_img = np.full((w+2, h+2, d), 255, dtype=np.uint8)

      # convert to 0-255 (resize want 0 255 img)
      if not self.custom_tranformations['rescale']:
        img = np.asarray(img*255, dtype=np.uint8)

      new_img[1:w+1, 1:h+1, :] = img 
      new_img = cv2.resize(new_img, (w, h))

      # reconvert to 0-1 (resize want 0 255 img)
      if not self.custom_tranformations['rescale']:
        new_img = data_preprocessing(new_img)

      if d == 1: new_img = np.expand_dims(new_img, axis=-1)

      return new_img

    @staticmethod
    def erode(img):
      w, h, d = img.shape
      rand = np.random.randint(1, 3)
      kernel = np.ones((rand, rand), np.uint8) 
      new_img = cv2.erode(img, kernel, iterations = 1)
      if d == 1: new_img = np.expand_dims(new_img, axis=-1)
      return new_img
#*********************************************************************************************************************************************



def contour_img(img):
  if not np.max(img) > 1: img = np.uint8(img*255)
  _3d = False
  
  if img.shape[2] == 3: _3d = True
  if _3d: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV) #THRESH_BINARY_INV

  # Run findContours - Note the RETR_EXTERNAL flag
  # Also, we want to find the best contour possible with CHAIN_APPROX_NONE
  contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

  # Create an output of all zeroes that has the same shape as the input
  # image
  img = np.zeros_like(img)

  # On this output, draw all of the contours that we have detected
  # in white, and set the thickness to be 1 pixel
  cv2.drawContours(img, contours, -1, 255, 1)

  if _3d:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return (255 - img) / 255. 
  else:
    return np.expand_dims( (255 - img) / 255. , axis=-1)

