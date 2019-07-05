import tensorflow as tf

# https://medium.com/apache-mxnet/sentiment-analysis-via-self-attention-with-mxnet-gluon-dc774d38ba69

class SelfAttentionLayer(tf.keras.layers.Layer):

    def __init__(self, attention_dim, **kwargs):
        self.init = tf.keras.initializers.RandomNormal()
        self.attention_dim = attention_dim
        super(SelfAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W1 = tf.Variable(self.init(shape=(input_shape[-1].value, self.attention_dim)), trainable=True, name='W1')
        self.W1 = tf.identity(self.W1)
        self.b = tf.Variable(self.init(shape=(self.attention_dim, )), trainable=True, name='b')
        self.b = tf.identity(self.b)
        self.W2 = tf.Variable(self.init(shape=(self.attention_dim, 1)), trainable=True, name='W2')
        self.W2 = tf.identity(self.W2)
        super(SelfAttentionLayer, self).build(input_shape)

    def call(self, hidden_states):
        d1 = tf.math.tanh(tf.nn.bias_add(tf.keras.backend.dot(hidden_states, self.W1), self.b))
        d2 = tf.keras.backend.dot(d1, self.W2)
        weights = tf.keras.layers.Softmax(axis=1)(d2)
        return weights

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class CustomReduceSumLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(CustomReduceSumLayer, self).__init__(**kwargs)

    def call(self, input):
        return tf.reduce_sum(input, axis=1)


def BdRNN_Attention(dropout=0.2, 
                    num_words=20000, 
                    emb_dim=128, 
                    max_len=100,
                    att_units=10,
                    emb_matrix=None,
                    trainable_flag=True):
    
    sequence_input = tf.keras.layers.Input(shape=(max_len,), dtype='int32')

    embedded_sequences = tf.keras.layers.Embedding(num_words,
                                                   emb_dim,
                                                   input_length=max_len,
                                                   trainable=trainable_flag)(sequence_input)

    # _h - hidden state outputs, _c - cell state outputs
    lstm, forward_h, _, backward_h, _ = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            units=64,
            dropout=0.2,
            return_sequences=True,
            return_state=True,
            recurrent_activation='relu',
            recurrent_initializer='glorot_uniform'
        )
    )(embedded_sequences)

    state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])

    # attention mechanism
    hidden_states_with_time_axis = tf.keras.layers.Reshape((1, state_h.shape[1]))(state_h)
    att_weights = SelfAttentionLayer(10)(hidden_states_with_time_axis)
    context_vec = tf.keras.layers.multiply([att_weights, lstm])
    context_vec = CustomReduceSumLayer()(context_vec)

    output = tf.keras.layers.Dense(6, activation='sigmoid')(context_vec)

    model = tf.keras.Model(inputs=sequence_input, outputs=output)
    adam = tf.keras.optimizers.Adam(lr=0.001, decay=0.01)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['acc'])

    return model