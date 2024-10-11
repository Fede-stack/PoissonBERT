import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Input, MultiHeadAttention, Dense, Attention
from tensorflow.keras.models import Model

class ReduceSumLayer(Layer):
    def __init__(self, axis, **kwargs):
        super(ReduceSumLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=self.axis)


class PoissonBERT(tf.keras.Model):
    def __init__(self, max_posts, embedding_dim, num_heads=3):
        super(PoissonBERT, self).__init__()
        self.max_posts = max_posts
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # Input layers
        self.input_reddit = Input((max_posts, embedding_dim))
        self.input_description = Input((embedding_dim,))
        self.inputs_choices = [Input((embedding_dim,)) for _ in range(4)]

        #Multi-head Attention Layer for Reddit posts
        self.memento_layers = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        
        # cross-Attention
        self.cross_attention_layer = Attention()

        # Choice Encoder + Cosine Similarity
        self.cosine_sim = tf.keras.layers.Dot(axes=-1, normalize=True)
        self.softmax = tf.keras.layers.Softmax(name='soft')

        # distributional Output layers
        self.dense_rate = Dense(1, activation='exponential')
        self.poisson_layer = tfp.layers.DistributionLambda(tfp.distributions.Poisson, name='poisson_layer')

        self.entropy_coefficient = tf.Variable(0.4, dtype=tf.float32, trainable=True)

    def call(self, inputs):
        input_reddit, input_description = inputs[0], inputs[1]
        inputs_choices = inputs[2:]

        # Repeat description for cross-attention
        repeated_description = tf.tile(tf.expand_dims(input_description, axis=1), [1, self.max_posts, 1])

        # Multi-head Attention for Reddit posts
        memento_posts, attention_weights = self.memento_layers(input_reddit, input_reddit, return_attention_scores=True)
        pooling_posts = ReduceSumLayer(axis = 1)(memento_posts)

        # Cross-Attention
        cross_attention_, cross_weights = self.cross_attention_layer([input_description, input_reddit, input_reddit], return_attention_scores=True)
        cross_attention = ReduceSumLayer(axis = 1)(cross_attention_)

        # Choice Encoder + Cosine Similarity
        output_similarity = [self.cosine_sim([cross_attention, inp_choice]) for inp_choice in inputs_choices]
        concatenated_similarities = tf.keras.layers.Concatenate()(output_similarity)
        softmax_similarities = self.softmax(concatenated_similarities)

        weighted_sum = tf.keras.layers.Add()([tf.keras.layers.Multiply()([inp_choice, softmax_similarities[:, i:i+1]]) for i, inp_choice in enumerate(inputs_choices)])
        
        merge_layer = pooling_posts + weighted_sum
        rate = self.dense_rate(merge_layer)
        p_y = self.poisson_layer(rate)

        return p_y, softmax_similarities

    def compile(self, optimizer, loss_weights):
        super(PoissonBERT, self).compile(
            optimizer=optimizer,
            loss={
                'poisson_layer': self.nll,
                'soft': self.custom_entropy_loss()
            },
            loss_weights=loss_weights
        )

    def nll(self, y_true, y_hat):
        return -y_hat.log_prob(y_true)

    def custom_entropy_loss(self):
        def loss(y_true, y_pred):
            entropy_loss = -tf.reduce_sum(y_pred * tf.math.log(y_pred + 1e-10), axis=-1)
            return entropy_loss
        return loss


model = PoissonBERT(max_posts, embedding_dim)
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
    loss_weights={'poisson_layer': 1-model.entropy_coefficient, 'soft': model.entropy_coefficient}
)

