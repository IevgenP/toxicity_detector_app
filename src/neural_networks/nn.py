import tensorflow as tf

# https://medium.com/apache-mxnet/sentiment-analysis-via-self-attention-with-mxnet-gluon-dc774d38ba69

class SelfAttentionLayer(tf.keras.layers.Layer):
    """Custom layer with self attention"""

    def __init__(self, attention_dim, **kwargs):
        """Initialize SelfAttentionLayer
        
        :param attention_dim: dimension of attention layer
        :type attention_dim: int
        """
        #self.init = tf.keras.initializers.RandomNormal()
        self.attention_dim = attention_dim
        super(SelfAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """Create trainable weight variable for the layer
        
        :param input_shape: shape of input tensor
        :type input_shape: int
        """
        assert len(input_shape) == 3
        self.W1 = self.add_weight(shape=(input_shape[-1].value, self.attention_dim),
                                  initializer='random_normal',
                                  trainable=True,
                                  name='W1')
        self.b = self.add_weight(shape=(self.attention_dim, ),
                                  initializer='zero',
                                  trainable=True,
                                  name='b')
        self.W2 = self.add_weight(shape=(self.attention_dim, 10),
                                  initializer='random_normal',
                                  trainable=True,
                                  name='W2')
        super(SelfAttentionLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, hidden_states, mask=None):
        """Executes logic of custom layer
        
        :param hidden_states: hidden states for all timesteps of preceeding layer
        :type hidden_states: tensor of floats
        :return: context vector which is a product of attention weights and hidden states
        :rtype: tensor of floats
        """
        print("Shape of W1 is {}, b is {}, W2 is {}".format(self.W1.shape, self.b.shape, self.W2.shape))
        d1 = tf.keras.backend.tanh(tf.keras.backend.bias_add(tf.keras.backend.dot(hidden_states, self.W1), self.b))
        d2 = tf.keras.backend.dot(d1, self.W2)
        print('--------d2', d2.shape)
        weights = tf.keras.layers.Softmax(axis=1)(d2)
        #weights = tf.keras.backend.max(weights, axis=2)
        #weights = tf.keras.backend.expand_dims(weights)
        print('---------weights after max', weights.shape)
        context_vec = tf.matmul(weights, hidden_states, transpose_a=True)
        # context_vec = tf.keras.backend.squeeze(context_vec, axis=1)
        print('---------context vector', context_vec.shape)
        return (context_vec, weights)

    def compute_output_shape(self, input_shape):
        """Helper function to specify the change in shape of the input when it passes through the layer
        
        :param input_shape: shape of input tensor
        :type input_shape: int
        :return: tuple with dimensions for output shape
        :rtype: int
        """
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        """Defines layer configuration
        
        :return: layer configurations
        :rtype: dictionary
        """
        config = {
            "attention_dim": self.attention_dim
        }
        base_config = super(SelfAttentionLayer, self).get_config()
        config.update(base_config)
        return config


def penalize_with_att_weights(y_true, y_pred, att_weights, batch_size):
    """Regularizer for loss based on attention weights
    
    :param y_true: true labels
    :type y_true: int / float
    :param y_pred: predicted probabilities of being classified as a label
    :type y_pred: float
    :param att_weights: tensor with attention weights
    :type att_weights: 
    :param batch_size: [description]
    :type batch_size: [type]
    :return: [description]
    :rtype: [type]
    """

    dot_product = tf.matmul(att_weights, att_weights, transpose_a=True)
    eye = tf.keras.backend.eye(dot_product.shape[1].value)
    diff = dot_product - eye
    reg = tf.reduce_sum(tf.square(diff)) / batch_size
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + 0.0001 * reg


#https://stackoverflow.com/a/45963039
def penalize_loss(att_weights, batch_size): 
    def penalization(y_true, y_pred):
        return penalize_with_att_weights(y_true, y_pred, att_weights, batch_size)
    return penalization
    

def BdRNN_Attention(dropout=0.4, 
                    num_words=20000, 
                    emb_dim=128, 
                    max_len=100,
                    att_units=10,
                    batch_size=128,
                    emb_matrix=None,
                    trainable_flag=True):
    """Neural network wrapped in function
    
    :param dropout: dropout fraction applied to different layers, defaults to 0.4
    :type dropout: float, optional
    :param num_words: max number of words in dictionary, defaults to 20000
    :type num_words: int, optional
    :param emb_dim: dimension of embeddings that represent words, defaults to 128
    :type emb_dim: int, optional
    :param max_len: number of tokens in each sample, defaults to 100
    :type max_len: int, optional
    :param att_units: dimenstion of first hidden layer in attention layer, defaults to 10
    :type att_units: int, optional
    :param emb_matrix: pre trained embedding matrix, defaults to None
    :type emb_matrix: numpy array, optional
    :param trainable_flag: select to train embeddings or not, defaults to True
    :type trainable_flag: bool, optional
    :return: model
    """
    
    sequence_input = tf.keras.layers.Input(shape=(max_len,), dtype='int32')

    embedded_sequences = tf.keras.layers.Embedding(num_words,
                                                   emb_dim,
                                                   weights=emb_matrix,
                                                   input_length=max_len,
                                                   trainable=trainable_flag)(sequence_input)

    emb_after_spatial_dr = tf.keras.layers.SpatialDropout1D(dropout)(embedded_sequences)

    gru = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            units=128,
            return_sequences=True
        )
    )(emb_after_spatial_dr)

    # attention mechanism
    context_vec, att_weights = SelfAttentionLayer(att_units)(gru)

    conv_context_2 = tf.keras.layers.Conv1D(filters=16, 
                                          kernel_size=2,
                                          padding='same',
                                          kernel_initializer = "glorot_uniform",
                                          activation='relu')(context_vec)
    avg_pool_2 = tf.keras.layers.GlobalAveragePooling1D()(conv_context_2)

    conv_context_3 = tf.keras.layers.Conv1D(filters=16, 
                                          kernel_size=3,
                                          padding='same',
                                          kernel_initializer = "glorot_uniform",
                                          activation='relu')(context_vec)
    avg_pool_3 = tf.keras.layers.GlobalAveragePooling1D()(conv_context_3)

    conv_context_5 = tf.keras.layers.Conv1D(filters=16, 
                                          kernel_size=5,
                                          padding='same',
                                          kernel_initializer = "glorot_uniform",
                                          activation='relu')(context_vec)
    avg_pool_5 = tf.keras.layers.GlobalAveragePooling1D()(conv_context_5)
    
    pooled = tf.keras.layers.Concatenate()([avg_pool_2, avg_pool_3, avg_pool_5])

    dense = tf.keras.layers.Dense(32, activation='relu')(pooled)
    output = tf.keras.layers.Dense(5, activation='sigmoid')(dense)

    model = tf.keras.Model(inputs=sequence_input, outputs=output)
    adam = tf.keras.optimizers.Adam(lr=0.001)
    #penalized_loss = penalize_loss(att_weights, batch_size)
    model.compile(loss='binary_crossentropy', # penalized_loss
                  optimizer=adam,
                  metrics=['acc'])

    return model


def RNN_CNN(dropout=0.4, 
            num_words=20000, 
            emb_dim=128, 
            max_len=100,
            att_units=10,
            emb_matrix=None,
            trainable_flag=True):
    sequence_input = tf.keras.layers.Input(shape=(max_len,), dtype='int32')
    x = tf.keras.layers.Embedding(num_words, emb_dim, weights=emb_matrix, trainable=trainable_flag)(sequence_input)
    x = tf.keras.layers.SpatialDropout1D(0.2)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x)
    x = tf.keras.layers.Conv1D(64, kernel_size = 3, padding = "same", kernel_initializer = "glorot_uniform", activation='relu')(x)
    avg_pool = tf.keras.layers.GlobalAvgPool1D()(x)
    max_pool = tf.keras.layers.GlobalMaxPool1D()(x)
    x = tf.keras.layers.Concatenate()([avg_pool, max_pool]) 
    output = tf.keras.layers.Dense(5, activation="sigmoid")(x)
    model = tf.keras.Model(sequence_input, output)
    model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(lr=1e-3),metrics=['acc'])
    return model