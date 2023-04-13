import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Input, LSTM, GRU, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, ZeroPadding3D, BatchNormalization, TimeDistributed, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.layers.core import Lambda

from utils.data_prep import *
prep = Data_Prep()
vocab_size = prep.char_to_num.vocabulary_size()+1

class LipNet(object):
    def __init__(self, inp_shape:tuple=(75, 46, 140, 1), epochs:int=100):
        self.inp_shape=inp_shape
        self.epochs=epochs
        

    def build(self, weight_path=None):
      model = Sequential()

      model.add(Conv3D(128, 3, input_shape=self.inp_shape, padding='same'))
      model.add(Activation('relu'))
      model.add(MaxPool3D((1,2,2)))

      model.add(Conv3D(256, 3, padding='same'))
      model.add(Activation('relu'))
      model.add(MaxPool3D((1,2,2)))

      model.add(Conv3D(75, 3, padding='same'))
      model.add(Activation('relu'))
      model.add(MaxPool3D((1,2,2)))

      model.add(TimeDistributed(Flatten()))

      model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
      model.add(Dropout(.5))
      model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
      model.add(Dropout(.5))

      model.add(Dense(vocab_size, kernel_initializer='he_normal', activation='softmax'))

      if weight_path != None:
        return model.load_weights(weight_path)
      else:
         return model

        

    def train(self, model, train_data, val_data, learn_rate:float, batch:int):
      def CTCLoss(y_true, y_pred):
          batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
          input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
          label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
          input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
          label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
          loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
          return loss

        
      
      model.compile(optimizer=Adam(learning_rate=learn_rate,  beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss=CTCLoss)
      
      checkpoint_dir = os.path.join('S:\\PROJECTS\\Lip_Reading\\checkpoints', "ckpt")
      checkpoint_callback = ModelCheckpoint(filepath=checkpoint_dir, monitor='loss', save_weights_only=True) 
      es_callback = EarlyStopping(monitor='val_loss', patience=10)


      history = model.fit(train_data, validation_data=val_data, verbose=1, epochs=self.epochs, callbacks=[checkpoint_callback])
      return history