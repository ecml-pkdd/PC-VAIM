# -*- coding: utf-8 -*-

import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, ActivityRegularization, Lambda, Add, concatenate
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse, mae

from tensorflow.keras.regularizers import l2, l1,L1L2
from tensorflow.keras.callbacks import History, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler,StandardScaler, QuantileTransformer, PowerTransformer
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, Activation

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape
from tensorflow_graphics.util import type_alias

from sklearn.model_selection import train_test_split

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.colors import LogNorm
import pylab as py

# -- set up the gpu env
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# -- main class of PC-VAIM
class PCVAIM():
    
    def __init__(self):
    
        # -- default parameters
        self.act='tanh'
        self.NUM_CLASSES = 1
        self.BATCH_SIZE = 64
        self.NUM_POINTS = 10
        self.latent_dim = 100
        self.input_shape = (1,) 
        self.epochs = 200
        self.l2_reg  = 1e-4
        self.DIR = 'outputs/'
        checkdir(self.DIR)
        self.history = History()
        
        # -- optimizer
        opt = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
      
        # -- create encoder and decoder models
        inputs = Input(shape = self.input_shape, name='encoder_input')  
        self.encoder = self.encoder(inputs)
        self.decoder = self.decoder()

        # -- decoder takes z and observables
        outputs = self.decoder(self.encoder(inputs)[2:4])
   
        # -- define and compile the model
        self.model = Model(inputs= inputs, outputs= outputs)
        self.model.compile(loss=[self.vae_loss, self.evaluate], optimizer=opt, experimental_run_tf_function=False )
        
    # -- build encoder model
    def encoder(self, inputs):

        aa    = tf.keras.Input(shape=(1,))
        x1_a  = Dense(512, activation=self.act,kernel_regularizer=l2(self.l2_reg))(aa)
        x2_a  = Dense(512, activation=self.act, kernel_regularizer=l2(self.l2_reg))(x1_a)
        x3_a  = Dense(512, activation=self.act,kernel_regularizer=l2(self.l2_reg))(x2_a)
        sc1_a = tf.keras.layers.Add()([x1_a,x3_a])
        x4_a  = Dense(512, activation=self.act,kernel_regularizer=l2(self.l2_reg))(sc1_a)
        x4    = Dense(512, activation=self.act,kernel_regularizer=l2(self.l2_reg))(x4_a)
        x4_   = Dense(self.NUM_POINTS*2, activation=self.act,kernel_regularizer=l2(self.l2_reg))(x4)
        y_x   = Reshape((self.NUM_POINTS,2))(x4_)

        self.z_mean    = Dense(self.latent_dim, name='z_mean')(x4_a)
        self.z_log_var = Dense(self.latent_dim, name='z_log_var')(x4_a)

        self.z = Lambda(self.sampling, output_shape=(self.latent_dim), name='z')([self.z_mean, self.z_log_var])

        encoder = Model(inputs=aa, outputs=[self.z_mean, self.z_log_var, self.z, y_x])
        encoder.summary()

        return encoder  
        
    # -- decoder model
    def decoder(self):

        en_latent      = Input(shape=(self.latent_dim))
        en_yx          =  Input(shape=(self.NUM_POINTS,2),name='yx')
        en_yx_out      = Lambda(lambda x: x)(en_yx)

        rt_0           = self.tnet(en_yx,en_yx.shape[2])
        r1             = self.conv_bn(rt_0, 512)
        r2             = self.conv_bn(r1, 1024)
        r3             = self.conv_bn(r2, 512)
        r_c1           = Add()([r1,r3])
        r6             = self.tnet(r_c1, 512)
        r7             = layers.GlobalMaxPooling1D()(r6)
        con_input      = concatenate([en_latent,r7],name = 'concat')

        r8             = Dense(1024, activation=self.act)(con_input)
        r9             = Dense(512, activation=self.act)(r8)
        r10            = Dense(1024, activation=self.act)(r9)
        r10            = Dense(1024, activation=self.act)(r10)

        r_params       = Dense(self.NUM_CLASSES, name = 'param_output')(r10)

        # instantiate decoder model
        decoder = Model(inputs=[en_latent,en_yx], outputs= [r_params, en_yx_out], name='decoder')
        decoder.summary()

        return decoder    
        
    # -- sampling function
    def sampling(self, args):
        
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.
        # Arguments:
            args (tensor): mean and log of variance of Q(z|X)
        # Returns:
            z (tensor): sampled latent vector
        """
        self.z_mean, self.z_log_var = args
        batch = K.shape(self.z_mean)[0]
        dim = K.int_shape(self.z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
            
        return self.z_mean + K.exp(0.5 * self.z_log_var) * epsilon


    def conv_bn(self, x, filters):
      x = layers.Conv1D(filters, kernel_size=1,strides=1,padding="valid")(x)
      #x = layers.BatchNormalization(momentum=0.0)(x)
      return layers.Activation(self.act)(x)

    def dense_bn(self, x, filters):
      x = layers.Dense(filters, kernel_regularizer=l2(self.l2_reg))(x)
    # x = layers.BatchNormalization(momentum=0.0)(x)
      return layers.Activation(self.act)(x)

    class OrthogonalRegularizer(tf.keras.regularizers.Regularizer):
      def __init__(self, num_features, l2reg=0.001):
          self.num_features = num_features
          self.l2reg = l2reg
          self.eye = tf.eye(num_features)

      def __call__(self, x):
          x = tf.reshape(x, (-1, self.num_features, self.num_features))
          xxt = tf.tensordot(x, x, axes=(2, 2))
          xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
          return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))
    
    def tnet(self, inputs, num_features):

        # Initalise bias as the indentity matrix
        bias = tf.keras.initializers.Constant(np.eye(num_features).flatten())
        reg = self.OrthogonalRegularizer(num_features)

        x = self.conv_bn(inputs, 512)
        x = self.conv_bn(x, 512)
        x = self.conv_bn(x, 512)
        x = layers.GlobalMaxPooling1D()(x)
        x = self.dense_bn(x, 256)
        x = self.dense_bn(x, 512)
        x = layers.Dense(
            num_features * num_features,
            kernel_initializer="zeros",
            bias_initializer=bias,
        )(x)
        feat_T = layers.Reshape((num_features, num_features))(x)
        # Apply affine transformation to input features
        return layers.Dot(axes=(2, 1))([inputs, feat_T])

    
    # KL and mae loss
    def vae_loss(self, inputs, outputs):
    
        mse_loss = mse(inputs, outputs)
        kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        kl_loss = K.mean(kl_loss, axis=-1)
        kl_loss *= -0.5
        loss = K.mean(kl_loss + mse_loss)
        return loss

    # -- chamfer distance adopted from tensorflow API
    @tf.function
    def evaluate(self, point_set_a: type_alias.TensorLike,
                point_set_b: type_alias.TensorLike,
                name: str = "chamfer_distance_evaluate") -> tf.Tensor:
      """Computes the Chamfer distance for the given two point sets.
      Note:
        This is a symmetric version of the Chamfer distance, calculated as the sum
        of the average minimum distance from point_set_a to point_set_b and vice
        versa.
        The average minimum distance from one point set to another is calculated as
        the average of the distances between the points in the first set and their
        closest point in the second set, and is thus not symmetrical.
      Note:
        This function returns the exact Chamfer distance and not an approximation.
      Note:
        In the following, A1 to An are optional batch dimensions, which must be
        broadcast compatible.
      Args:
        point_set_a: A tensor of shape `[A1, ..., An, N, D]`, where the last axis
          represents points in a D dimensional space.
        point_set_b: A tensor of shape `[A1, ..., An, M, D]`, where the last axis
          represents points in a D dimensional space.
        name: A name for this op. Defaults to "chamfer_distance_evaluate".
      Returns:
        A tensor of shape `[A1, ..., An]` storing the chamfer distance between the
        two point sets.
      Raises:
        ValueError: if the shape of `point_set_a`, `point_set_b` is not supported.
      """
      with tf.name_scope(name):
        point_set_a = tf.convert_to_tensor(value=point_set_a)
        point_set_b = tf.convert_to_tensor(value=point_set_b)

        shape.compare_batch_dimensions(
            tensors=(point_set_a, point_set_b),
            tensor_names=("point_set_a", "point_set_b"),
            last_axes=-3,
            broadcast_compatible=True)
        # Verify that the last axis of the tensors has the same dimension.
        dimension = point_set_a.shape.as_list()[-1]

        # Create N x M matrix where the entry i,j corresponds to ai - bj (vector of
        # dimension D).
        difference = (
            tf.expand_dims(point_set_a, axis=-2) -
            tf.expand_dims(point_set_b, axis=-3))
        # Calculate the square distances between each two points: |ai - bj|^2.
        square_distances = tf.einsum("...i,...i->...", difference, difference)

        minimum_square_distance_a_to_b = tf.reduce_min(
            input_tensor=square_distances, axis=-1)
        minimum_square_distance_b_to_a = tf.reduce_min(
            input_tensor=square_distances, axis=-2)

        return tf.math.reduce_sum(
            tf.reduce_mean(input_tensor=minimum_square_distance_a_to_b, axis=-1) +
            tf.reduce_mean(input_tensor=minimum_square_distance_b_to_a, axis=-1))
        
    # -- Train the model
    def train(self, X_train, y_train):
            
        checkpointer = ModelCheckpoint(filepath = self.DIR, verbose=1, save_best_only=True)
        self.model.fit(x = y_train, y = [y_train,X_train], shuffle=True, epochs= self.epochs,
                       batch_size= 64,validation_split=0.2,callbacks = [checkpointer, self.history])
        return self.history

    
 # -- check if direcorty exists, otherwise create one
def checkdir(path):
    if not os.path.exists(path): 
        os.makedirs(path)

# -- generate data samples
def generate_data(N_samples = 4000, N_points=10):

    x_data = np.empty([0, N_points, 1])
    a_data = np.empty([0, 1])
    for i in range(N_samples):
        a = np.random.uniform(-2, 2, [1, 1])
        x = np.random.uniform(-1, 1,[1, N_points, 1])
        x_data = np.concatenate([x_data, x], axis = 0)
        a_data = np.concatenate([a_data, a], axis = 0)
        
    fx= (a_data*x_data[:,:,0])**2
    fx_x_pointClouds= np.dstack([fx,x_data])
    X_train, X_test, y_train, y_test = train_test_split(fx_x_pointClouds, a_data, test_size=0.20)

    return X_train, X_test, y_train, y_test

# -- get latent variables
def predict (model, y_train):

  latent_mean   = model.encoder.predict([y_train])[0]
  latent_logvar = model.encoder.predict([y_train])[1]
  z             = model.encoder.predict([y_train])[2]

  latent_var = np.exp(latent_logvar)
  latent_std = np.sqrt(latent_var)

  return latent_mean, latent_std, z

# -- sampling function
def sample(mean, std, y_train):  
  
  latent_dim=100
  SAMPLE_SIZE = mean.shape[0]
  z_samples = np.empty([SAMPLE_SIZE, latent_dim])
  LATENT_SAMPLE_SIZE = y_train.shape[0]

  for i in range(0,SAMPLE_SIZE):
      for j in range(0,latent_dim):
          z_samples[i,j] = np.random.normal(mean[i%LATENT_SAMPLE_SIZE, j], std[i%LATENT_SAMPLE_SIZE,j])

  return z_samples

# -- create a test example
def generate_test_example(z):

    N_samples= 1
    N_points=10
    x_data = np.empty([0, N_points, 1])
    a_data = np.empty([0, 1])
    for i in range(N_samples):
        a = np.random.uniform(1, 1, [1, 1])
        x = np.random.uniform(-1, 1,[1, N_points, 1])
        x_data = np.concatenate([x_data, x], axis = 0)
        a_data = np.concatenate([a_data, a], axis = 0)
    fx  = (a_data[:,:]*x_data[:,:,0])**2
    fx  = np.dstack([fx])
    pcl = np.dstack([fx,x_data])
    pcl = np.repeat(pcl,z.shape[0],axis=0)

    return pcl
