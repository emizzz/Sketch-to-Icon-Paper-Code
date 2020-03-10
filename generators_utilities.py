import random
import numpy as np
import tensorflow as tf

#*********************************************************************************************************************************************
'''
inspired by https://github.com/ArkaJU/Sketch-Retrieval---Siamese 
'''
class SiameseMultiDomainDataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_x1, data_x2, out_with_true_class=False, n_classes=10, shuffle=False):
        self.x1, self.y1 = data_x1
        self.x2, self.y2 = data_x2
        self.x1, self.y1, self.x2, self.y2 = self.x1.tolist(), self.y1.tolist(), self.x2.tolist(), self.y2.tolist()

        self.shuffle = shuffle
        self.out_with_true_class = out_with_true_class
 
        self._reset_data()

        #minimum number of examples in the smaller class
        self.iter_per_epoch = min([len(i) for i in self.x1_indexes])-1


    def __len__(self):
        'Denotes the number of batches per epoch'
        
        return 1

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
          self.x1, self.y1 = shuffle_with_same_indexes(self.x1, self.y1)
          self.x2, self.y2 = shuffle_with_same_indexes(self.x2, self.y2)
        
        self._reset_data()

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate data
        return self.__data_generation()

    def __data_generation(self):
        labels = []
        data = []

        if self.out_with_true_class:
          true_labels = []

        for _ in range(self.iter_per_epoch): 
          label = None
            
          # create balanced data
          if np.random.uniform() >= 0.5:
            
            old_label = np.random.randint(len(self.x1_indexes))
            x1_radn_idx = np.random.choice(self.x1_indexes[old_label])
            x2_radn_idx = np.random.choice(self.x2_indexes[old_label])

            curr_x1 = self.x1[x1_radn_idx]
            curr_x2 = self.x2[x2_radn_idx]

            self.x1_indexes[old_label].remove(x1_radn_idx)
            self.x2_indexes[old_label].remove(x2_radn_idx)

            label = 0. #same domain
            if self.out_with_true_class: true_label = old_label
            
          else:
            list_labels = [i for i, _ in enumerate(self.x1_indexes)]
            
            old_label = np.random.choice(list_labels)
            list_labels.remove(old_label)
            new_label = np.random.choice(list_labels)
            x1_radn_idx = np.random.choice(self.x1_indexes[old_label])
            x2_radn_idx = np.random.choice(self.x2_indexes[new_label])

            curr_x1 = self.x1[x1_radn_idx]
            curr_x2 = self.x2[x2_radn_idx]

            self.x1_indexes[old_label].remove(x1_radn_idx)
            self.x2_indexes[new_label].remove(x2_radn_idx)

            label = 1. #diff domain
            if self.out_with_true_class: true_label = old_label


          labels.append(label)
          data.append(np.array([curr_x1, curr_x2]))
          if self.out_with_true_class:
            true_labels.append(true_label)

        if self.out_with_true_class:
          return {'input_1': np.asarray(data)[:, 0, :, :, :], 'input_2': np.asarray(data)[:, 1, :, :, :]}, {'output_1': np.array(labels), 'output_2': np.array(true_labels)}
          #return np.array(data), {'output_1': np.array(labels), 'output_2': np.array(true_labels)}
        else:
          return np.array(data), np.array(labels)

    
    def _reset_data(self):
      self.x1_indexes = [[] for arr in set(self.y1)]
      self.x2_indexes = [[] for arr in set(self.y1)]

      for i, l in enumerate(self.y1):
        self.x1_indexes[l].append(i)
      for i, l in enumerate(self.y2):
        self.x2_indexes[l].append(i)
#*********************************************************************************************************************************************




#*********************************************************************************************************************************************
class TripletDataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x_anchors, y_anchors, x_images, y_images, batch_size=128, dim=(28,28,1), shuffle=True, transf_anchor_func=None, transf_image_func=None):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        
        temp_anchors = [(a, y_anchors[i]) for i, a in enumerate(x_anchors)]
        temp_images = [(a, y_images[i]) for i, a in enumerate(x_images)]
        self.anchors = np.asarray(sorted(temp_anchors, key = lambda x: x[1]))
        self.images = np.asarray(sorted(temp_images, key = lambda x: x[1]))
        self.transf_anchor_func = transf_anchor_func
        self.transf_image_func = transf_image_func
        assert np.array_equal(self.anchors[:, 1], self.images[:, 1])

        self.shuffle = shuffle
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.anchors) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        b_from = index*self.batch_size      #batch start index
        b_to = (index+1)*self.batch_size    #batch end index

        # Generate data
        res = self.__data_generation(b_from, b_to)

        return res

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
          self.anchors, self.images = self.random_shifting(self.anchors, self.images)

    def __data_generation(self, b_from, b_to):

        x_anchors = self.anchors[b_from: b_to]
        x_images = self.images[b_from: b_to]
        
        anch = np.zeros([self.batch_size, self.dim[0], self.dim[1], self.dim[2]],dtype=np.float32)
        imgs = np.zeros([self.batch_size, self.dim[0], self.dim[1], self.dim[2]],dtype=np.float32)
        labels = np.zeros([self.batch_size, 1], dtype=np.uint8)
        
        for i in range(self.batch_size):
          anch[i] = x_anchors[i][0] if not self.transf_anchor_func else self.transf_anchor_func(x_anchors[i][0])
          imgs[i] =  x_images[i][0] if not self.transf_image_func else self.transf_image_func(x_images[i][0])
          labels[i] = x_images[i][1]

        return [anch, imgs], labels
        #return {'input_1': anch, 'input_2': imgs}, {'output_1': labels}


    def random_shifting(self, anchors, images):

      p = np.random.permutation(len(images[:, 1]))
      images = images[p]
      anchors = anchors[p]

      class_indexes = {}
      [class_indexes.setdefault(y[1], []).append(i) for i, y in enumerate(images)]

      new_class_indexes = {c: sorted(indexes, key=lambda k: random.random()) for c, indexes in class_indexes.items()}
      
      for i, (_, y) in enumerate(images):
          idx = new_class_indexes[y].pop(0)
          anchors[idx] = (anchors[i][0], anchors[i][1])
      
      return anchors, images
#*********************************************************************************************************************************************




