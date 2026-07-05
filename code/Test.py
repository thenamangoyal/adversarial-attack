from PIL import Image
import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib.slim.nets import inception, resnet_v2
import csv
from collections import OrderedDict
slim = tf.contrib.slim

tf.flags.DEFINE_string( 'input_path', '', 'The original images path.')
tf.flags.DEFINE_string('checkpoint_path', '', 'Path to checkpoint for inception network.')
tf.flags.DEFINE_string('modified_path', '', 'Path to the modified images.')
tf.flags.DEFINE_integer('image_width', 299, 'Width of each input images.')
tf.flags.DEFINE_integer('image_height', 299, 'Height of each input images.')
tf.flags.DEFINE_integer('batch_size', 16, 'How many images process at one time.')
tf.flags.DEFINE_string('model','inception_v3', 'The model to be tested.')

FLAGS = tf.flags.FLAGS

input_path=FLAGS.input_path
modified_path=FLAGS.modified_path
checkpoints_dir=FLAGS.checkpoint_path

def load_images(input_dir, Filenames, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in Filenames:
    with tf.gfile.Open(os.path.join(input_dir,filepath+'.png')) as f:
      image = np.array(Image.open(f).convert('RGB').resize(batch_shape[1:-1])).astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image * 2.0 - 1.0
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images

class Model(object):
  """Model class for CleverHans library."""
  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False
  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    if FLAGS.model == "inception_v3":
        with slim.arg_scope(inception.inception_v3_arg_scope()):
          _, end_points = inception.inception_v3(
              x_input, num_classes=self.num_classes, is_training=False,
              reuse=reuse)
        output = end_points['Predictions']
    elif FLAGS.model == "inception_v2":
        with slim.arg_scope(inception.inception_v2_arg_scope()):
          _, end_points = inception.inception_v2(
              x_input, num_classes=self.num_classes, is_training=False,
              reuse=reuse)
        output = end_points['Predictions']
    elif FLAGS.model == "resnet_v2_152":
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
          net, end_points = resnet_v2.resnet_v2_152(
              x_input, num_classes=self.num_classes, is_training=False,
              reuse= reuse)
        output = end_points['predictions']
    else:
        raise ValueError("The model should be either inception_v3, inception_v2 or resnet_v2_152")
    self.built = True
    # Strip off the extra reshape op at the output
    probs = output.op.inputs[0]
    return probs

batch_shape=[FLAGS.batch_size,FLAGS.image_width,FLAGS.image_height,3]

num_classes = 1001

tf.logging.set_verbosity(tf.logging.INFO)




with open('dev_dataset.csv','r') as f:
  T=csv.DictReader(f, delimiter=',')
  Filenames=np.array([row['ImageId'] for row in T])
  
no_files = len(Filenames)
I=np.zeros(no_files).astype('bool')
F=tf.gfile.Glob(os.path.join(modified_path, '*.png'))
F=[F[i][len(modified_path)+1:-4] for i in range(len(F))]
I=np.ones(no_files).astype('bool')
for i in range(no_files):
  if not Filenames[i] in F:
    I[i]=False
Filenames=Filenames[I==True]
no_files=len(Filenames)

print(no_files)
init_label=np.zeros(no_files).astype(int)
modified_label=np.zeros(no_files).astype(int)
modified_Prob=np.zeros(no_files).astype('float32')
init_Prob=np.zeros(no_files).astype('float32')
modified_init_Prob=np.zeros(no_files).astype('float32')

with tf.Graph().as_default():
  # Prepare graph
  x_input = tf.placeholder(tf.float32, shape=batch_shape)
  with tf.Session() as sess:
    model=Model(num_classes)
    prob=model(x_input)
    
    #print slim.get_model_variables()
    saver=tf.train.Saver(slim.get_model_variables())
    saver.restore(sess,os.path.join(checkpoints_dir, FLAGS.model+ ".ckpt"))
    it=0
    for filenames, images in load_images(input_path, Filenames, batch_shape):
      P=np.array(sess.run(prob,{x_input:images.astype('float32')}))
      init_label[it:min(no_files,it+batch_shape[0])]=np.argmax(P[:min(no_files-it,batch_shape[0])],1)
      init_Prob[it:min(no_files,it+batch_shape[0])]=np.max(P[:min(no_files-it,batch_shape[0])],1)
      it+=batch_shape[0]
      print(it)
    it = 0
    for filenames, images in load_images(modified_path, Filenames, batch_shape):
      P=np.array(sess.run(prob,{x_input:images.astype('float32')}))
      modified_label[it:min(no_files,it+batch_shape[0])]=np.argmax(P[:min(no_files-it,batch_shape[0])],1)
      modified_Prob[it:min(no_files,it+batch_shape[0])]=np.max(P[:min(no_files-it,batch_shape[0])],1)
      for i in range(min(no_files-it,batch_shape[0])):
        modified_init_Prob[it+i]=P[i,init_label[it+i]]
      it+=batch_shape[0]
      print(it)

with open('dev_dataset.csv','r') as f:
  T=csv.DictReader(f, delimiter=',')
  Label=np.array([int(row['TrueLabel']) for row in T])
  #print Label
Label=Label[I==True]

import time;

loc = time.localtime(time.time())
with open('logs/'+str(loc.tm_year)+'_'+str(loc.tm_mon)+'_'+str(loc.tm_mday)+'_'+str(loc.tm_hour)+'_'+str(loc.tm_min)+'_'+str(loc.tm_sec)+'.csv','w') as f:
  W=csv.DictWriter(f,fieldnames=OrderedDict([('Intial label',None), ('Initial Prob',None), ('Modified Label',None), ('Modified Prob',None), ('Modified init Prob',None), ('Label',None)]))
  W.writeheader()
  for i in range(no_files):
    W.writerow({'Intial label':init_label[i], 'Initial Prob':init_Prob[i],'Modified Label':modified_label[i] ,'Modified Prob':modified_Prob[i],'Modified init Prob':modified_init_Prob[i],'Label':Label[i]})

print('Initial accuracy:',np.sum(init_label==Label).astype('float32')/no_files,'\n')
print('Final accuracy:',np.sum(modified_label==Label).astype('float32')/no_files,'\n')
