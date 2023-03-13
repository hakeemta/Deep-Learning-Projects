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
        
        # V(s)
        value_out = layers.Dense(1, name='value_out')(context)

        # Base layers
        self.base_model = models.Model(obs_input, [context, value_out],
                                         name='base_model')
        self.base_model.summary()

        # Autoregressive action sampler
        ctx_input = layers.Input(shape=(num_outputs,), name='ctx_input')

        # P(a1 | obs)
        from_logits = layers.Dense(n_towers, name='from_logits')(ctx_input)
        self.from_model = models.Model(ctx_input, from_logits, 
                                         name='from_model')
        self.from_model.summary()

        # P(a2 | a1)
        # --note: typically you'd want to implement P(a2 | a1, obs) as follows:
        from_input = layers.Input(shape=(n_towers, ), name='from_input')
        to_hidden = layers.Dense(n_hiddens, name='to_hidden',
                                          activation='relu')(from_input)
        to_logits = layers.Dense(n_towers, name='to_logits')(to_hidden)
        self.to_model = models.Model(from_input, to_logits,
                                       name='to_model')
        self.to_model.summary()

    def forward(self, input_dict, state, seq_lens):
        context, self._value_out = self.base_model(input_dict['obs'])
        return context, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
    

MODEL_NAME = 'autoreg_model'
ModelCatalog.register_custom_model(MODEL_NAME, AutoregressiveModel)
