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
        self.bb = tf.Variable(self.init(shape=(self.attention_dim, )), trainable=True, name='bb')
        self.bb = tf.identity(self.bb)
        self.W2 = tf.Variable(self.init(shape=(self.attention_dim, 1)), trainable=True, name='W2')
        self.W2 = tf.identity(self.W2)
        super(SelfAttentionLayer, self).build(input_shape)

    def call(self, hidden_states):
        d1 = tf.math.tanh(tf.nn.bias_add(tf.keras.backend.dot(hidden_states, self.W1), self.bb))
        d2 = tf.keras.backend.dot(d1, self.W2)
        weights = tf.keras.layers.Softmax(axis=1)(d2)
        context_vec = tf.matmul(weights, hidden_states, transpose_a=True)
        
        return context_vec

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = {
            "attention_dim": self.attention_dim
        }
        base_config = super(SelfAttentionLayer, self).get_config()
        config.update(base_config)
        return config


def BdRNN_Attention(dropout=0.4, 
                    num_words=20000, 
                    emb_dim=128, 
                    max_len=100,
                    att_units=10,
                    emb_matrix=None,
                    trainable_flag=True):
    
    sequence_input = tf.keras.layers.Input(shape=(max_len,), dtype='int32')

    embedded_sequences = tf.keras.layers.Embedding(num_words,
                                                   emb_dim,
                                                   weights=emb_matrix,
                                                   input_length=max_len,
                                                   trainable=trainable_flag)(sequence_input)

    emb_after_spatial_dr = tf.keras.layers.SpatialDropout1D(dropout)(embedded_sequences)

    # _h - hidden state outputs, _c - cell state outputs
    gru = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            units=128,
            return_sequences=True
        )
    )(emb_after_spatial_dr)

    # attention mechanism
    context_vec = SelfAttentionLayer(att_units)(gru)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(context_vec)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(context_vec)
    pooled = tf.keras.layers.Concatenate()([avg_pool, max_pool])

    #dense_1 = tf.keras.layers.Dense(256, activation='relu')(context_vec)
    output = tf.keras.layers.Dense(6, activation='sigmoid')(pooled)

    model = tf.keras.Model(inputs=sequence_input, outputs=output)
    adam = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['acc'])

    return model