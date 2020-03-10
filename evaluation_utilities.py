#*********************************************************************************************************************************************
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from data_utilities import show_img
from IPython.display import HTML
from google.colab.output import eval_js
from base64 import b64decode
import PIL
from io import BytesIO
import matplotlib.cm as cm
from matplotlib.pyplot import figure, imshow, axis
from pylab import rcParams


rcParams['figure.figsize'] = 10, 10

#*********************************************************************************************************************************************
'''
  get the n most similar images
'''
class RankImages():
  def __init__(self, _X, _model, _model_target=None, _n=10, _show_im=False, _show_dist=False):
        self.cache = []
        self._X = _X
        self._n = _n
        self._show_im = _show_im
        self._show_dist = _show_dist

        if not _model_target:
          self._model_target = _model
        else:
          self._model_target = _model_target

        self._model = _model

        
  #is better to pass an entire batch for the cache
  def get_n_most_similar_images(self, _target, _returnType=None):

      target_embed = np.squeeze(self._model_target.predict(_target[np.newaxis, ...]))
      res = []

      if len(self.cache) < 1:
        for x in self._X:
          image_feat = np.squeeze(self._model.predict(x[np.newaxis, ...]))
          self.cache.append((x, image_feat)) 

      for idx, (img, img_embed) in enumerate(self.cache):
        res.append((idx, img, np.linalg.norm( target_embed - img_embed)))
        
      res.sort(key=lambda x: x[2])
      res = res[:self._n]

      if self._show_im:
        for i in res:
          if self._show_dist:
            print("dist: ", i[2])
          show_img(i[2]) 

      
      if _returnType == 'img':
        return np.asarray(res)[:, 1]
      elif _returnType == 'indexes':        
        return np.asarray(res)[:, 0]
      else:
        return None

  @staticmethod
  def format_result(query, data):
    data.insert(0, np.clip(query, 1, 1))
    data.insert(0, query)
    len_data = len(data)

    
    fig = figure()
    for i in range(len_data):
      im = data[i]
      a = fig.add_subplot(1,len_data,i+1)
      
      if im.shape[2] == 1: im = im[:, :, 0]
      imshow(im, cmap='Greys_r')
      axis('off')
#*********************************************************************************************************************************************

  

#*********************************************************************************************************************************************
'''
  KNN
  if we doesn't have a classification model, but instead we have a model that produces an embedding,
  this method allows us to predict the embedding class using the KNN method.

  USAGE:
    knn = KNN(x_train, y_train, k=100)      => train the knn alg
    y_test_pred = knn.get_labels(x_test)    => predict the class (rapresented by the most frequent class in the nearests K examples from the target)

'''
class KNN:
  def __init__(self, _X_train, _y_train, k=100):
    print("knn inizializing...")

    if len(_X_train.shape) > 2:
        _X_train = self.reshape_data(_X_train)
      
    self.knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    self.knn.fit(_X_train, _y_train)
    print("knn end training")

  def get_labels(self, _X_test):
    print("knn predicting...")

    if len(_X_test.shape) > 2:
      _X_test = self.reshape_data(_X_test)

    pred =  self.knn.predict(_X_test)
    print("knn end prediction")
    return pred

  def reshape_data(self, data):
    try:
      return [d.reshape(d.shape[0]*d.shape[1]) for d in data[:, :, :, 0]]
    except:
      print("cannot handle this data shape")
      return data

#*********************************************************************************************************************************************



#*********************************************************************************************************************************************
'''
  GET_SCORE
  compute:
    - f1_score
    - precision
    - recall

'''
def get_score(_y_test_true, _y_test_pred):
  computed_f1_score = f1_score(_y_test_true, _y_test_pred, average='micro')
  print("f1_score:", computed_f1_score)
  return computed_f1_score
#*********************************************************************************************************************************************



#*********************************************************************************************************************************************
# thanks to https://gist.github.com/korakot/8409b3feec20f159d8a50b0a811d3bca
def paint_brush_img(w=28, h=28, line_width=15, preprocessed=False, show=False):

  canvas_html = """
  <canvas width=%d height=%d></canvas>
  <button>Finish</button>
  <script>
  var canvas = document.querySelector('canvas')
  var ctx = canvas.getContext('2d')
  ctx.lineWidth = %d
  var button = document.querySelector('button')
  var mouse = {x: 0, y: 0}
  canvas.addEventListener('mousemove', function(e) {
    mouse.x = e.pageX - this.offsetLeft
    mouse.y = e.pageY - this.offsetTop
  })
  canvas.onmousedown = ()=>{
    ctx.beginPath()
    ctx.moveTo(mouse.x, mouse.y)
    canvas.addEventListener('mousemove', onPaint)
  }
  canvas.onmouseup = ()=>{
    canvas.removeEventListener('mousemove', onPaint)
  }
  var onPaint = ()=>{
    ctx.lineTo(mouse.x, mouse.y)
    ctx.stroke()
  }
  var data = new Promise(resolve=>{
    button.onclick = ()=>{
      resolve(canvas.toDataURL('image/png'))
    }
  })
  </script>
  """

  display(HTML(canvas_html % (300, 300, line_width)))
  data = eval_js("data")
  binary = b64decode(data.split(',')[1])

  image = PIL.Image.open(BytesIO(binary))
  image = image.resize((w, h))
  image = np.asarray(image)
  image = 255 - image[:, :, 3]    # from rgba to grayscale (Assumptions: Your image is always black-n-white; White is defined as a transparency;)
  
  if preprocessed:
    image = np.expand_dims(image, axis=-1)
    image = image / 255.
    image = image.astype(np.float32)

  if show:
    show_img(image)

  return image
#*********************************************************************************************************************************************




