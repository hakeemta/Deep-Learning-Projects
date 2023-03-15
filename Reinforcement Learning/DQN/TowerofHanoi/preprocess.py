import tensorflow as tf
import numpy as np


class Dataset(tf.keras.utils.Sequence):
    def __init__(self,path, batch_size=64):
        self.data = np.load(path, mmap_mode=None)
        self.batch_size = batch_size
                
    def __len__(self):
        return len(self.data) //self.batch_size
            
    def __getitem__(self, idx):
        batch_x = self.data[idx * self.batch_size :(idx + 1) *self.batch_size]
        batch_y = self.data[idx * self.batch_size :(idx + 1) *self.batch_size]#.flatten()
        batch_y = tf.one_hot(indices=batch_y, depth=11, on_value=1.0, off_value=0.0, axis=-1).numpy()
        return batch_x, batch_y.reshape(self.batch_size, -1)
    
    
    
    
class Encoder(tf.keras.Model):
    def __init__(self, name='None'):
        super().__init__()
        self.num_classes = 330
        self.model = tf.keras.Sequential([
                    tf.keras.layers.Dense(256, activation='softplus'),
                    tf.keras.layers.Dropout(0.10),
                    tf.keras.layers.Dense(128, activation='hard_sigmoid'),
                    tf.keras.layers.Dropout(0.15),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dropout(0.15),
                    tf.keras.layers.Dense(32, activation='softplus'),
                    tf.keras.layers.Dense(16, activation='softplus')
                    ])
    
    def call(self, x):
        preprocess = tf.one_hot(indices=x, depth=11, on_value=1.0, off_value=0.0, axis=-1)
        output = self.model(tf.keras.layers.Reshape((330,), input_shape=preprocess.shape)(preprocess))
    
        return output





def decoder(num_class=330):
    return tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='hard_sigmoid'),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(64, activation='softplus'),
            tf.keras.layers.Dropout(0.10),
            tf.keras.layers.Dense(128, activation='hard_sigmoid'),
            tf.keras.layers.Dropout(0.10),
            tf.keras.layers.Dense(256, activation='softplus'),
            tf.keras.layers.Dense(num_class, activation='relu')
        ])
    
    
    
    
class AutoEncoder(tf.keras.Model):
    def __init__(self, name='None'):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = decoder(num_class=330)
    
    def call(self, x):
        encoder_hidden = self.encoder(x)
        output = self.decoder(encoder_hidden)
        return output
    