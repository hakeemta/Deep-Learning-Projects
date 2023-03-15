import tensorflow as tf
import numpy as np


class Dataset(tf.keras.utils.Sequence):
    def __init__(self,path, batch_size=64):
        self.data = np.load(path, mmap_mode=None)
        self.batch_size = batch_size
                
    def __len__(self):
        return len(self.data) //self.batch_size
            
    def __getitem__(self, idx):
        batch = self.data[idx * self.batch_size :(idx + 1) *self.batch_size]
        batch = tf.one_hot(indices=batch, depth=7, on_value=1.0, off_value=0.0, axis=-1).numpy()
        return batch.reshape(self.batch_size, -1), batch.reshape(self.batch_size, -1)
    
    
    
    
class Encoder(tf.keras.Model):
    def __init__(self, name='None'):
        super().__init__()
        self.num_classes = 126
        self.model = tf.keras.Sequential([
                    #tf.keras.layers.Dense(256, activation='softplus'),
                    #tf.keras.layers.Dropout(0.10),
                    tf.keras.layers.Dense(64, activation='hard_sigmoid'),
                    #tf.keras.layers.Dropout(0.15),
                    tf.keras.layers.Dense(32, activation='softplus'),
                    #tf.keras.layers.Dropout(0.15),
                    tf.keras.layers.Dense(16, activation='softplus'),
                    tf.keras.layers.Dense(8, activation='softplus')
                    ])
    
    def call(self, x):
        output = self.model(x)

    
        return output





def decoder(num_class=126):
    return tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='hard_sigmoid'),
            #tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(32, activation='softplus'),
            #tf.keras.layers.Dropout(0.10),
            tf.keras.layers.Dense(64, activation='hard_sigmoid'),
            #tf.keras.layers.Dropout(0.10),
            #tf.keras.layers.Dense(256, activation='softplus'),
            tf.keras.layers.Dense(num_class, activation='relu')
        ])
    
    
    
    
class AutoEncoder(tf.keras.Model):
    def __init__(self, name='None'):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = decoder(num_class=126)
    
    def call(self, x):
        encoder_hidden = self.encoder(x)
        output = self.decoder(encoder_hidden)
        return output
    
    
