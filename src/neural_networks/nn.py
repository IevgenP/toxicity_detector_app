import tensorflow as tf

# https://medium.com/apache-mxnet/sentiment-analysis-via-self-attention-with-mxnet-gluon-dc774d38ba69

class SelfAttentionLayer(tf.keras.layers.Layer):
    """Custom layer with self attention"""

    def __init__(self, attention_dim, **kwargs):
        """Initialize SelfAttentionLayer
        
        :param attention_dim: dimension of attention layer
        :type attention_dim: int
        """
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
        self.W2 = self.add_weight(shape=(self.attention_dim, 1),
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
        d1 = tf.keras.backend.tanh(tf.keras.backend.bias_add(tf.keras.backend.dot(hidden_states, self.W1), self.b))
        d2 = tf.keras.backend.dot(d1, self.W2)
        weights = tf.keras.layers.Softmax(axis=1)(d2)
        context_vec = tf.matmul(weights, hidden_states, transpose_a=True)
        context_vec = tf.keras.backend.sum(context_vec, axis=1)
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


def BdRNN_HA(dropout=0.4, 
             num_words=20000, 
             emb_dim=128,
             max_sentence_length=100, 
             max_sentences=15,
             att_units=10,
             batch_size=None,
             emb_matrix=None,
             trainable_flag=True):
    """RNN with hierarchical attention
    
    :param dropout: dropout fraction applied to different layers, defaults to 0.4
    :type dropout: float, optional
    :param num_words: max number of words in dictionary, defaults to 20000
    :type num_words: int, optional
    :param emb_dim: dimension of embeddings that represent words, defaults to 128
    :type emb_dim: int, optional
    :param max_sentence_length: maximum number of words in a sentence, defaults to 100
    :type max_sentence_length: int, optional
    :param max_sentences: maximum number of sentences in one sample, defaults to 15
    :type max_sentences: int, optional
    :param att_units: dimenstion of first hidden layer in attention layer, defaults to 10
    :type att_units: int, optional
    :param batch_size: size of batch fed to neral network during training, defaults to None
    :type batch_size: int, optional
    :param emb_matrix: pre-trained embedding matrix, defaults to None
    :type emb_matrix: numpy array, optional
    :param trainable_flag: boolean that defines whether embeddings should be trained or not, defaults to True
    :type trainable_flag: bool, optional
    :return: model
    """
    
    input_sentence = tf.keras.layers.Input(shape=(max_sentence_length, ), dtype='int32')
    embedded_sequences = tf.keras.layers.Embedding(
        num_words,
        emb_dim,
        weights=emb_matrix,
        input_length=max_sentence_length,
        trainable=trainable_flag
    )(input_sentence)
    gru_sentence = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=128, return_sequences=True))(embedded_sequences)
    att_sentence, weights_of_words = SelfAttentionLayer(attention_dim=att_units)(gru_sentence)
    SentenceEncoder = tf.keras.models.Model(input_sentence, att_sentence)

    input_sample = tf.keras.layers.Input(shape=(max_sentences, max_sentence_length), dtype='int32')
    dense_sample = tf.keras.layers.TimeDistributed(SentenceEncoder)(input_sample)
    gru_sample = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=128, return_sequences=True))(dense_sample)
    att_sample, weights_of_sentences = SelfAttentionLayer(attention_dim=att_units)(gru_sample)
    output = tf.keras.layers.Dense(units=5, activation='sigmoid')(att_sample)
    model = tf.keras.models.Model(input_sample, output)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    return model