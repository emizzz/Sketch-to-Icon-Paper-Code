import tensorflow as tf
from matplotlib import pyplot as plt
from IPython.display import clear_output
from matplotlib import pyplot
from scipy.spatial import distance_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np




#*********************************************************************************************************************************************
class SimpleEmbeddingNet(tf.keras.Model):

  def __init__(self):
    self.filter_size = 24
    super(SimpleEmbeddingNet, self).__init__()

    self.l1_conv    = tf.keras.layers.Conv2D(self.filter_size,kernel_size=3,activation='relu',input_shape=(28,28,1))
    self.l1_batch   = tf.keras.layers.BatchNormalization()
    self.l2_conv    = tf.keras.layers.Conv2D(self.filter_size,kernel_size=3,activation='relu')
    self.l2_batch   = tf.keras.layers.BatchNormalization()
    self.l3_conv    = tf.keras.layers.Conv2D(self.filter_size,kernel_size=5,strides=2,padding='same',activation='relu')
    self.l3_batch   = tf.keras.layers.BatchNormalization()
    self.l3_dropout = tf.keras.layers.Dropout(0.4)
    self.l4_conv    = tf.keras.layers.Conv2D(self.filter_size*2,kernel_size=3,activation='relu')
    self.l4_batch   = tf.keras.layers.BatchNormalization()
    self.l5_conv    = tf.keras.layers.Conv2D(self.filter_size*2,kernel_size=3,activation='relu')
    self.l5_batch   = tf.keras.layers.BatchNormalization()
    self.l6_conv    = tf.keras.layers.Conv2D(self.filter_size*2,kernel_size=5,strides=2,padding='same',activation='relu')
    self.l6_batch   = tf.keras.layers.BatchNormalization()
    self.l6_dropout = tf.keras.layers.Dropout(0.4)
    self.l7_flatten = tf.keras.layers.Flatten()
    self.l7_dense   = tf.keras.layers.Dense(128, activation='relu')

  
  def call(self, x):
    x = self.l1_conv(x)
    x = self.l1_batch(x)
    x = self.l2_conv(x)
    x = self.l2_batch(x)
    x = self.l3_conv(x)
    x = self.l3_batch(x)
    x = self.l3_dropout(x)
    x = self.l4_conv(x)
    x = self.l4_batch(x)
    x = self.l5_conv(x)
    x = self.l5_batch(x)
    x = self.l6_conv(x)
    x = self.l6_batch(x)
    x = self.l6_dropout(x)
    x = self.l7_flatten(x)
    x = self.l7_dense(x)   

    return x


class SimpleNet(tf.keras.Model):

  def __init__(self, n_classes=10):
    super(SimpleNet, self).__init__()

    self.embedding_net = SimpleEmbeddingNet()

    self.l7_batch   = tf.keras.layers.BatchNormalization()
    self.l7_dropout = tf.keras.layers.Dropout(0.4)
    self.l8_dense   = tf.keras.layers.Dense(n_classes, activation='softmax')

  def call(self, x):
    
    x = self.embedding_net.call(x) 

    x = self.l7_batch(x)
    x = self.l7_dropout(x)
    x = self.l8_dense(x)

    return x
#*********************************************************************************************************************************************




#*********************************************************************************************************************************************
# thanks to: https://gist.github.com/stared/dfb4dfaf6d9a8501cd1cc8b8cb806d2e

class PlotLosses(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        
        clear_output(wait=True)
        
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.legend()
        
        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend()
        
        plt.show();
        
#*********************************************************************************************************************************************


#*********************************************************************************************************************************************
net_callbacks = [
  PlotLosses(),
  tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto', restore_best_weights=True)
]
print("net_callbacks var created")
#*********************************************************************************************************************************************


#*********************************************************************************************************************************************
class AutoencoderViz(tf.keras.callbacks.Callback):
    def __init__(self):
      pass
    def on_epoch_end(self, batch, logs={}):

      rand = np.random.randint(0, len(x_test), 1)[0]
      target_1 = self.model.predict(x_test[rand][np.newaxis, ...])
      show_img(target_1[0, :, :, 0])
#*********************************************************************************************************************************************


#*********************************************************************************************************************************************
# cut network layers
# example: viz_net = CutNet(net.encoder, input_shape=(1, 28, 28, 1), n_layers=4, verbose=True)

class CutNet(tf.keras.Model):

  def __init__(self, _net, input_shape=(1, 28, 28, 1), n_layers=float('inf'), verbose=False):
    super(CutNet, self).__init__()
    self._layers = []

    for i, _layer in enumerate(_net._layers):
      if i < n_layers:
        self._layers.append(_layer)

    for i, _layer in enumerate(self._layers):
      setattr(self, _layer.name, _layer)

    self.build(input_shape)
    if verbose: self.summary()

  def call(self, x):
    for _layer in self._layers: 
      x = _layer(x) 
    return x
#*********************************************************************************************************************************************



#*********************************************************************************************************************************************
# visualize feature maps
# example: visualize_feature_maps(viz_net, x_test[10]*255)

def visualize_feature_maps(net, img, square=8):
  img = np.expand_dims(img, axis=0)
  print(net.layers)
  layer_name = net.layers[-1].name
  feature_maps = net.predict(img)
  ii = 0
  for fmap in feature_maps:
    if "conv" in layer_name:
      print(layer_name, fmap.shape)

      for filt_i in range(len(fmap[0, 0, :])):
        # specify subplot and turn of axis
        ii = ii+1
        ax = pyplot.subplot(square, square, ii)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale

        pyplot.imshow(fmap[:, :, filt_i], cmap='gray')
      # show the figure
  pyplot.show()
#*********************************************************************************************************************************************


#*********************************************************************************************************************************************
'''
with this function you can transfer the weights from same architecture networks that has coded in different ways
'''
def load_nested_net_weights(_model_from, _model_to):

  '''
  this function return the leafs if a neural network,
  '''
  def search_leafs(model):
    leafs = []  

    def _search_leafs(_model):
      for l in _model.layers:
        
        if not hasattr(l, 'layers'):
          leafs.append(l)
        else:
          _search_leafs(l)

    _search_leafs(model)
    return leafs

  _model_from_layers = search_leafs(_model_from)
  _model_to_layers = search_leafs(_model_to)

  for i, l_to in enumerate(_model_from_layers):
    _model_to_layers[i].set_weights(l_to.get_weights())

  print("weights loaded")
#*********************************************************************************************************************************************



#*********************************************************************************************************************************************

class VectorVisualizator:
  def __init__(self):
    plt.rcParams['figure.figsize'] = [10, 10]

  def visualize(self, data, labels):
    pca = PCA(n_components=2)
    data = StandardScaler().fit_transform(data)
    pc = pca.fit_transform(data)
        
    for i in range(len(pc)):
        x = pc[i][0]
        y = pc[i][1]
        plt.plot(x, y, '') # 'bo'
        plt.annotate(labels[i], (x, y))

    plt.show()
#*********************************************************************************************************************************************



#*********************************************************************************************************************************************
'''
validation data is needed because the generator is None inside the callback:
https://github.com/keras-team/keras/issues/10472

callbacks=[PredictionCallback([x_test_draws, y_test_draws], every_n_ep=5)],
'''
class PredictionCallback(tf.keras.callbacks.Callback):   
  def __init__(self, val_data, batch_size = 1, every_n_ep = 1):
        super().__init__()
        self.every_n_ep = every_n_ep
        self.validation_data = val_data[0]                                          #only draws
        self.validation_labels = np.asarray(val_data[1])
        self.viz = VectorVisualizator()

        

  def on_epoch_end(self, epoch, logs={}):
        self.predicted_data = self.model.draws_net.predict(self.validation_data)    #only draws
        print("ep", epoch)
        #if epoch % 4 == self.every_n_ep:
        self.viz.visualize(self.predicted_data, self.validation_labels)
        return
#*********************************************************************************************************************************************




