import tensorflow as tf
from tensorflow.keras import layers, models

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models import ModelCatalog

from ray.rllib.utils.framework import try_import_tf
tf1, tf, tfv = try_import_tf()

# Source: https://github.com/ray-project/ray/blob/master/rllib/examples/models/autoregressive_action_model.py

class AttentionScore(layers.Attention):
    def __init__(self, use_scale=False, score_mode='dot', **kwargs) -> None:
        super(AttentionScore, self).__init__(use_scale, score_mode, **kwargs)

    def call(self,
             inputs,
             mask=None,
             training=None):
        self._validate_call_args(inputs=inputs, mask=mask)
        q = inputs[0]
        v = inputs[1]
        k = inputs[2] if len(inputs) > 2 else v
        # q_mask = mask[0] if mask else None

        v_mask = mask[1] if mask else None
        scores = self._calculate_scores(query=q, key=k)

        if v_mask is not None:
            # Mask of shape [batch_size, 1, Tv].
            v_mask = tf.expand_dims(v_mask, axis=-2)

        # @TODO: Bello et al. (2016): clip the result
        # (before masking!) within [−C; C] (C = 10) using tanh
        # scores = 10.0 * tf.math.tanh(scores)

        scores_mask = v_mask
        if scores_mask is not None:
            padding_mask = tf.logical_not(scores_mask)
            # Bias so padding positions do not contribute to attention distribution.
            # Note 65504. is the max float16 value.
            if scores.dtype is tf.float16:
                scores -= 65504. * tf.cast(padding_mask, dtype=scores.dtype)
            else:
                scores -= 1.e9 * tf.cast(padding_mask, dtype=scores.dtype)

        return scores



class AutoregressiveModel(TFModelV2):
    def __init__(self, obs_space, action_space, 
                 num_outputs, model_config, name, n_hiddens=16,):
        
        super().__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        n_towers = action_space[0].n

        # Inputs
        obs_input = layers.Input(shape=obs_space.shape, name='obs_input')
        # flattened = layers.Flatten() (obs_input)

        embeddings = layers.Dense(n_hiddens, name='embeddings1',
                               activation='relu')(obs_input)
        embeddings = layers.Dense(n_hiddens, name='embeddings2',
                               activation='relu')(embeddings)

        # V(s)
        context_v = layers.Dense(n_hiddens, name='context1_v',
                               activation='relu')(obs_input)
        context_v = layers.Dense(n_hiddens, name='context2_v',
                               activation='relu')(context_v)
        
        context_v = layers.Flatten() (context_v)
        value_out = layers.Dense(1, name='value_out')(context_v)

        # Base layers
        self.base_model = models.Model(obs_input, [embeddings, value_out],
                                       name='base_model')
        self.base_model.summary()

        # Autoregressive action sampler
        mask_input = layers.Input(shape=(n_towers, ), 
                                  name='mask_input', dtype=tf.bool)
        ctx_input = layers.Input(shape=(1, n_hiddens), 
                                 name='ctx_input')
        emb_input = layers.Input(shape=(n_towers, n_hiddens), 
                                 name='emb_input')

        logits = AttentionScore() ([ctx_input, emb_input], 
                                   mask=[None, mask_input])
        
        self.selection_model = models.Model([emb_input, ctx_input, mask_input],
                                      logits, name='action')

    def select(self, embeddings, mask):
        mask = tf.squeeze(mask)
        return self.selection_model([embeddings, self.context_value, mask])

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']
        embeddings, self._value_out = self.base_model(obs)
        

        context = tf.reduce_mean(embeddings, axis=1)
        self.context_value = tf.expand_dims(context, axis=1)

        mask = tf.reduce_any(obs > 0, axis=-1)
        mask = tf.cast(mask, dtype=embeddings.dtype)
        mask = tf.expand_dims(mask, axis=-1)
        
        packed = tf.concat([embeddings, mask], axis=-1)
        return packed, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
    

MODEL_NAME = 'autoreg_model'
ModelCatalog.register_custom_model(MODEL_NAME, AutoregressiveModel)
