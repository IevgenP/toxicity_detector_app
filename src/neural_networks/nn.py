import tensorflow as tf


class CustomReduceSumLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(CustomReduceSumLayer, self).__init__(**kwargs)

    def call(self, input):
        return tf.reduce_sum(input, axis=1)


def BdRNN_Attention(dropout=0.2, 
                    num_words=20000, 
                    emb_dim=128, 
                    max_len=100,
                    linear_lstm_dropout=0.2,
                    att_units=10,
                    emb_matrix=None,
                    trainable_flag=False):
    
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
    attention_hidden_layer = tf.keras.layers.Dense(att_units)(hidden_states_with_time_axis)
    attention_hidden_layer_tanh = tf.keras.layers.Dense(att_units, activation='tanh')(attention_hidden_layer)
    score = tf.keras.layers.Dense(1)(attention_hidden_layer_tanh)
    
    attention_weights = tf.keras.layers.Softmax(axis=1)(score)
    context_vector = tf.keras.layers.multiply([attention_weights, lstm])
    context_vector_reduced = CustomReduceSumLayer()(context_vector)

    output = tf.keras.layers.Dense(6, activation='sigmoid')(context_vector_reduced)

    model = tf.keras.Model(inputs=sequence_input, outputs=output)
    adam = tf.keras.optimizers.Adam(lr=0.01, decay=0.01)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['acc'])

    return model