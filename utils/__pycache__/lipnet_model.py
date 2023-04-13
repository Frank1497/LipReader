from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv3D,  Input, LSTM, GRU, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, ZeroPadding3D, BatchNormalization, TimeDistributed, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from utils.data_prep import *
prep = Data_Prep()
vocab_size = prep.char_to_num.vocabulary_size()+1

class LipNet(object):
    def __init__(self, inp_shape:tuple=(75, 46, 140, 1), epochs:int=100):
        self.inp_shape=inp_shape
        self.epochs=epochs

    def build(self):
        model = Sequential()
        model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))
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

        model.add(Dense(prep.char_to_num.vocabulary_size()+1, kernel_initializer='he_normal', activation='softmax'))
        return model
    

    
    
    
    

    def train(self, model, train_data, val_data, learn_rate:float, batch:int):
        """
        adaptive_lr, ctcloss and produce example are special loss functions 
        that are used when fitting and training the model
        """

        def scheduler(epoch, lr):
            if epoch < 30:
                return lr
            else:
                return lr * tf.math.exp(-0.1)


        def CTCLoss(y_true, y_pred):
            batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
            input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
            label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

            input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
            label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

            loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
            return loss
       
       
        class ProduceExample(tf.keras.callbacks.Callback): 
            def __init__(self, dataset) -> None: 
                self.dataset = dataset.as_numpy_iterator()
            
            def on_epoch_end(self, epoch, logs=None) -> None:
                data = self.dataset.next()
                yhat = self.model.predict(data[0])
                decoded = tf.keras.backend.ctc_decode(yhat, [75,75], greedy=False)[0][0].numpy()
                for x in range(len(yhat)):           
                    print('Original:', tf.strings.reduce_join(prep.num_to_char(data[1][x])).numpy().decode('utf-8'))
                    print('Prediction:', tf.strings.reduce_join(prep.num_to_char(decoded[x])).numpy().decode('utf-8'))
                    print('~'*100)





        model.compile(optimizer=Adam(learning_rate=0.0001), loss=CTCLoss)
        checkpoint_callback = ModelCheckpoint(filepath='/content/drive/MyDrive/Lip_reading/checkpoint/model_{epoch:02d}.h5', monitor='loss', save_weights_only=True) 
        schedule_callback = LearningRateScheduler(self.adaptive_lr)
        example_callback = ProduceExample(val_data)
        es_callback = EarlyStopping(monitor='val_loss', patience=10)


        history = model.fit(train_data, validation_data=val_data, verbose=1, epochs=self.epochs, callbacks=[checkpoint_callback])
        return history