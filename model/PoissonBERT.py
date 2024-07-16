# Inputs
Input_Reddit = Input((MAX_POSTS, embedding_dim))
Input_Description = Input((embedding_dim, ))
Repeated_Description = tf.tile(tf.expand_dims(Input_Description, axis=1), [1, MAX_POSTS, 1])

#Multi-head Attention Layer for Reddit posts
Memento_Layers = MultiHeadAttention(num_heads=3, key_dim=embedding_dim)
Memento_Posts, attention_weights = Memento_Layers(Input_Reddit, Input_Reddit, return_attention_scores=True)

#Pooling from MHA
Pooling_Posts = tf.reduce_sum(Memento_Posts, axis=1)

#Cross-Attention
cross_attention_layer = Attention()
cross_attention_, cross_weights = cross_attention_layer([Input_Description, Input_Reddit, Input_Reddit], return_attention_scores=True)
cross_attention = tf.reduce_sum(cross_attention_, axis=1)

#Choice Encoder + Cosine Similarity implementation
Inputs_Choices = []
Output_Similarity = []
for i in range(4):
    inp_choice_i = Input((embedding_dim, ))
    Inputs_Choices.append(inp_choice_i)
    Cosine_Sim = tf.keras.layers.Dot(axes=-1, normalize=True)([cross_attention, inp_choice_i])
    Output_Similarity.append(Cosine_Sim)

Concatenated_Similarities = tf.keras.layers.Concatenate()(Output_Similarity)
Softmax_Similarities = tf.keras.layers.Softmax(name='soft')(Concatenated_Similarities)
entropy_coefficient = tf.Variable(0.4, dtype=tf.float32, trainable=True)
# entropy_coefficient = Dense(1, activation = 'sigmoid')(Input_Description)

Weighted_Sum = tf.keras.layers.Add()([tf.keras.layers.Multiply()([inp_choice, Softmax_Similarities[:, i:i+1]]) for i, inp_choice in enumerate(Inputs_Choices)])

Merge_Layer = Pooling_Posts + Weighted_Sum

rate = Dense(1, activation='exponential')(Merge_Layer)
p_y = tfp.layers.DistributionLambda(tfp.distributions.Poisson, name='poisson_layer')(rate)

#model definition
PoissonBERT = Model(inputs=[Input_Reddit, Input_Description] + Inputs_Choices, outputs=[p_y, Softmax_Similarities])

def NLL(y_true, y_hat):
  #Poisson Negative log-likelihood implementation
    return -y_hat.log_prob(y_true)

def custom_entropy_loss():
  #Negative Entropy implementation
    def loss(y_true, y_pred):
        entropy_loss = -tf.reduce_sum(y_pred * tf.math.log(y_pred + 1e-10), axis=-1)
        return entropy_loss
    return loss
# negloglik = lambda y, rv_y: -rv_y.log_prob(y)

PoissonBERT.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                    loss={'poisson_layer': NLL, 'soft': custom_entropy_loss()},
                    loss_weights={'poisson_layer': 1-entropy_coefficient, 'soft': entropy_coefficient})
