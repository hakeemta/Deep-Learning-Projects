from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_action_dist import Categorical, ActionDistribution
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()


# Source: https://github.com/ray-project/ray/blob/master/rllib/examples/models/autoregressive_action_dist.py

class AutoregDistribution(ActionDistribution):
    """Action distribution P(_from, to) = P(_from) * P(to | _from)"""

    def __init__(self, inputs, model):
        inputs, mask = tf.split(inputs, [-1, 3], axis=-1)
        self.mask = mask

        super().__init__(inputs, model)

    def deterministic_sample(self):
        # First, sample _from.
        from_dist = self._from_distribution()
        _from = from_dist.deterministic_sample()

        # Sample to conditioned on _from.
        to_dist = self._to_distribution(_from)
        to = to_dist.deterministic_sample()
        self._action_logp = from_dist.logp(_from) + to_dist.logp(to)

        return tf.stack([_from, to], axis=1)

    def sample(self):
        # First, sample _from.
        from_dist = self._from_distribution()
        _from = from_dist.sample()

        # Sample to conditioned on _from.
        to_dist = self._to_distribution(_from)
        to = to_dist.sample()
        self._action_logp = from_dist.logp(_from) + to_dist.logp(to)
        
        return tf.stack([_from, to], axis=1)

    def logp(self, actions):
        _from, to = actions[:, 0], actions[:, 1]
        from_dist = self._from_distribution()
        to_dist = self._to_distribution(_from)
        return from_dist.logp(_from) + to_dist.logp(to)

    def sampled_action_logp(self):
        return self._action_logp

    def entropy(self):
        from_dist = self._from_distribution()
        to_dist = self._to_distribution(from_dist.sample())
        return from_dist.entropy() + to_dist.entropy()

    def kl(self, other):
        from_dist = self._from_distribution()
        from_terms = from_dist.kl(other._from_distribution())

        _from = from_dist.sample()
        to_terms = self._to_distribution(_from).kl(other._to_distribution(_from))
        return from_terms + to_terms

    def _from_distribution(self):
        # from_logits = self.model.from_model(self.inputs)
        from_logits, _ = tf.split(self.inputs, 2, axis=-1)
        
        # Mask empty towers
        mask = tf.cast(self.mask, dtype=tf.bool)
        mask = tf.logical_not(mask) # Inverse the masking

        from_logits -= (1.e9 * tf.cast(mask, dtype=tf.float32)) 
        
        from_dist = Categorical(from_logits)
        return from_dist

    def _to_distribution(self, _from):
        # from_encoded = tf.one_hot(_from, 3, 
        #                           on_value=1, off_value=0) 
        # to_logits = self.model.to_model(self.inputs)
        _, to_logits = tf.split(self.inputs, 2, axis=-1)
        
        # Mask from logits in to_logits
        mask = tf.one_hot(_from, 3,
                          on_value=1, off_value=0, 
                          dtype=tf.float64)
        to_logits -=  (1.e9 * tf.cast(mask, 
                                      dtype=tf.float32))

        to_dist = Categorical(to_logits)
        return to_dist

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        # controls model output feature vector size
        shape = (3, model_config['custom_model_config']['n_hiddens'] + 1)
        return shape
    

ACTION_DIST_NAME = 'autoreg_dist'
ModelCatalog.register_custom_action_dist(ACTION_DIST_NAME,
                                        AutoregDistribution)
