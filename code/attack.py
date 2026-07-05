"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from cleverhans.attacks import Attack, FastGradientMethod, BasicIterativeMethod, VirtualAdversarialMethod
from cleverhans.model import Model, CallableModelWrapper
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception

slim = tf.contrib.slim


tf.flags.DEFINE_string( 'master', '', 'The address of the TensorFlow master to use.')
tf.flags.DEFINE_string('checkpoint_path', '', 'Path to checkpoint for inception network.')
tf.flags.DEFINE_string('input_dir', '', 'Input directory with images.')
tf.flags.DEFINE_string('output_dir', '', 'Output directory with images.')
tf.flags.DEFINE_float('max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')
tf.flags.DEFINE_integer('image_width', 299, 'Width of each input images.')
tf.flags.DEFINE_integer('image_height', 299, 'Height of each input images.')
tf.flags.DEFINE_integer('batch_size', 16, 'How many images process at one time.')
tf.flags.DEFINE_string('method', 'BinIM', 'The method of attack.')

FLAGS = tf.flags.FLAGS

def load_images(input_dir, batch_shape):
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
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    with tf.gfile.Open(filepath) as f:
      image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0
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


def save_images(images, filenames, output_dir):
  """Saves images to the output directory.

  Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
  """
  for i, filename in enumerate(filenames):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # so rescale them back to [0, 1].
    with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
      img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
      Image.fromarray(img).save(f, format='PNG')


class InceptionModel(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    with slim.arg_scope(inception.inception_v3_arg_scope()):
      _, end_points = inception.inception_v3(
          x_input, num_classes=self.num_classes, is_training=False,
          reuse=reuse)
    self.built = True
    output = end_points['Predictions']
    # Strip off the extra reshape op at the output
    probs = output.op.inputs[0]
    return probs

class BinaryIterativeMethod(Attack):

    def __init__(self, model, back='tf', sess=None):
        super(BinaryIterativeMethod, self).__init__(model, back, sess)
        self.feedable_kwargs = {'eps': np.float32,
                                'eps_iter': np.float32,
                                'y': np.float32,
                                'y_target': np.float32,
                                'clip_min': np.float32,
                                'clip_max': np.float32}
        self.structural_kwargs = ['ord', 'nb_iter']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'probs')

    def generate(self, x, **kwargs):
        import tensorflow as tf

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        # Initialize loop variables
        eta = 0

        # Fix labels to the first model predictions for loss computation
        model_preds = self.model.get_probs(x)
        preds_max = tf.reduce_max(model_preds, 1, keep_dims=True)
        if self.y_target is not None:
            y = self.y_target
            targeted = True
        elif self.y is not None:
            y = self.y
            targeted = False
        else:
            y = tf.to_float(tf.equal(model_preds, preds_max))
            y = tf.stop_gradient(y)
            targeted = False

        y_kwarg = 'y_target' if targeted else 'y'
        fgm_params = { y_kwarg: y, 'ord': self.ord,
                      'clip_min': self.clip_min, 'clip_max': self.clip_max}

        self.eps_iter = self.eps/ 2
        for i in range(self.nb_iter):
            FGM = FastGradientMethod(self.model, back=self.back,
                                     sess=self.sess)
            # Compute this step's perturbation
            eta = FGM.generate(x + eta, eps= self.eps_iter,  **fgm_params) - x
            
            self.eps_iter/= 2
            # Clipping perturbation eta to self.ord norm ball
            if self.ord == np.inf:
                eta = tf.clip_by_value(eta, -self.eps, self.eps)
            elif self.ord in [1, 2]:
                reduc_ind = list(xrange(1, len(eta.get_shape())))
                if self.ord == 1:
                    norm = tf.reduce_sum(tf.abs(eta),
                                         reduction_indices=reduc_ind,
                                         keep_dims=True)
                elif self.ord == 2:
                    norm = tf.sqrt(tf.reduce_sum(tf.square(eta),
                                                 reduction_indices=reduc_ind,
                                                 keep_dims=True))
                #eta = eta * self.eps / norm

        # Define adversarial example (and clip if necessary)
        adv_x = x + eta
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        return adv_x

    def parse_params(self, eps=0.3, nb_iter=10, y=None,
                     ord=np.inf, clip_min=None, clip_max=None,
                     y_target=None, **kwargs):
        # Save attack-specific parameters
        self.eps = eps
        self.nb_iter = nb_iter
        self.y = y
        self.y_target = y_target
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        return True



def main(_):
  # Images for inception classifier are normalized to be in [-1, 1] interval,
  # eps is a difference between pixels so it should be in [0, 2] interval.
  # Renormalizing epsilon from [0, 255] to [0, 2].
  eps = 2.0 * FLAGS.max_epsilon / 255.0
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  num_classes = 1001

  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)

    model = InceptionModel(num_classes)
    if(FLAGS.method == "BinIM"):
      attack = BinaryIterativeMethod(model)
      args = {'x': x_input, 'eps': eps, 'nb_iter': 10,'clip_min': -1., 'clip_max': 1.}
    elif(FLAGS.method == "FGM"):
      attack = FastGradientMethod(model)
      args = {'x': x_input, 'eps': eps,'clip_min': -1., 'clip_max': 1.}
    elif(FLAGS.method == "BIM"):
      attack = BasicIterativeMethod(model)
      args = {'x': x_input, 'eps': eps, 'eps_iter': eps/ 10, 'nb_iter': 10,'clip_min': -1., 'clip_max': 1.}
    elif(FLAGS.method == "VAM"):
      attack = VirtualAdversarialMethod(model)
      args = {'x': x_input, 'eps': eps, 'xi': 1e-5, 'eps_iter': eps/ 10, 'nb_iter': 10,'clip_min': -1., 'clip_max': 1.}
    else:
      raise ValueError("The method should be either FGM, BIM, VAM or BinIM")

    x_adv = attack.generate(**args)

    # Run computation
    saver = tf.train.Saver(slim.get_model_variables())
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=tf.train.Scaffold(saver=saver),
        checkpoint_filename_with_path=FLAGS.checkpoint_path,
        master=FLAGS.master)

    it=0
    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
      for filenames, images in load_images(FLAGS.input_dir, batch_shape):
        adv_images = sess.run(x_adv, feed_dict={x_input: images})
        save_images(adv_images, filenames, FLAGS.output_dir)
        it+=1
        print (it*batch_shape[0])


if __name__ == '__main__':
  tf.app.run()
