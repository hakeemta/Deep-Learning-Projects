import tensorflow as tf
from tensorflow.keras import layers, models

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models import ModelCatalog

from ray.rllib.utils.framework import try_import_tf
tf1, tf, tfv = try_import_tf()

# Source: https://github.com/ray-project/ray/blob/master/rllib/examples/models/autoregressive_action_model.py

class AutoregressiveModel(TFModelV2):
    def __init__(self, obs_space, action_space, 
                 num_outputs, model_config, name, n_hiddens=16,):
        
        super().__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        n_towers = action_space[0].n

        # Inputs
        obs_input = layers.Input(shape=obs_space.shape, name='obs_input')
        flattened = layers.Flatten() (obs_input)

        context = layers.Dense(n_hiddens, name='context1',
                               activation='relu')(flattened)
        context = layers.Dense(n_hiddens, name='context2',
                               activation='relu')(context)
        
        logits = layers.Dense(2 * n_towers, name='logits') (context)
                
        # V(s)
        context_v = layers.Dense(n_hiddens, name='context1_v',
                               activation='relu')(flattened)
        context_v = layers.Dense(n_hiddens, name='context2_v',
                               activation='relu')(context_v)
        value_out = layers.Dense(1, name='value_out')(context_v)

        # Base layers
        self.base_model = models.Model(obs_input, [logits, value_out],
                                       name='base_model')
        self.base_model.summary()

        # Autoregressive action sampler
        # ctx_input = layers.Input(shape=(n_hiddens, ), name='ctx_input')

        # P(a1 | obs)
        # from_logits = layers.Dense(n_towers, name='from_logits')(ctx_input)
        # self.from_model = models.Model(ctx_input, from_logits, 
        #                                  name='from_model')
        # self.from_model.summary()

        # P(a2 | a1)
        # from_input = layers.Input(shape=(n_towers, ), name='from_input')
        # from_embedding = layers.Dense(n_hiddens, name='from_embedding',
        #                               activation='relu')(from_input)
        # combined = tf.keras.layers.Concatenate(axis=1) ([ctx_aggr, from_embedding])
        
        # to_logits = layers.Dense(n_towers, name='to_logits')(ctx_input)
        # self.to_model = models.Model(ctx_input, to_logits,
        #                              name='to_model')
        # self.to_model.summary()

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']
        logits, self._value_out = self.base_model(obs)

        mask = tf.reduce_any(obs > 0, axis=-1)
        mask = tf.cast(mask, dtype=logits.dtype)
        
        packed = tf.concat([logits, mask], axis=-1)
        return packed, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
    

MODEL_NAME = 'autoreg_model'
ModelCatalog.register_custom_model(MODEL_NAME, AutoregressiveModel)
