from data_preprocessor import *
import tensorflow as tf


class Seq2SeqModel:
    def __init__(self, data, hyperparams):
        self.data = data
        self.params = hyperparams
        
        self.model_inputs()
        
        #Lungimea maxima a unei secvente
        self.sequence_length = tf.placeholder_with_default(self.params['seq_length'], None,  name = 'sequence_length')
        
    def get_predictions(self):
        self.encoder_rnn()
        self.decoder_rnn()
        
        return self.training_predictions, self.test_predictions
    
    def model_inputs(self):
        """
        Creeaza containere speciale (neinitializate cu date)
            inputs : date de input
            targets : outputul tinta pentru input
            learning_rate : rata de invatare (rata de miscare catre minimul functii de cost)
            keep_prob : sansa unui neuron de a fi activ in timpul executiei (reduce overfiting)
       """
        
        self.inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
        self.targets = tf.placeholder(tf.int32, [None, None], name = 'target')
        
        self.learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')
        self.keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
        
    def preprocess_targets(self):
        """
        Preprocesarea datelor tinta (adaugarea simbolului `inceput de propozitie` la inceputul datelor tinta/raspunsurilor)
        """
        left_side = tf.fill([self.params['batch_size'], 1], self.data.questions_words2int['<SOS>'])  
        right_side = tf.strided_slice(self.targets, [0, 0], [self.params['batch_size'], -1], [1, 1])
    
        return tf.concat([left_side, right_side], axis = 1)
    
    def create_multi_rnn_cell(self):
        lstm = tf.contrib.rnn.BasicLSTMCell(self.params['rnn_size'])
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = self.keep_prob)    
    
        return tf.contrib.rnn.MultiRNNCell([lstm_dropout] * self.params['num_layers'])
            
    
    def encoder_rnn(self):
        forward_cell = self.create_multi_rnn_cell()
        backward_cell = self.create_multi_rnn_cell()
        
        encoder_embedded_input = tf.contrib.layers.embed_sequence(self.inputs,
                                                              len(self.data.answers_words2int) + 1,
                                                              self.params['encoding_embedding_size'],
                                                              initializer = tf.random_uniform_initializer(0, 1))
        
        encoder_output, self.encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = forward_cell,
                                                                        cell_bw = backward_cell,
                                                                        sequence_length = self.sequence_length,
                                                                        inputs  = encoder_embedded_input,
                                                                        dtype = tf.float32)
        
    def decode_training_set(self, decoder_cell, decoder_embedded_input, decoding_scope, output_function):
        attention_states = tf.zeros([self.params['batch_size'], 1, decoder_cell.output_size])
        
        attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states,
                                                                                                                                        attention_option = 'bahdanau',
                                                                                                                                        num_units = decoder_cell.output_size )
    
        training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(self.encoder_state[0],
                                                                          attention_keys,
                                                                          attention_values,
                                                                          attention_score_function,
                                                                          attention_construct_function,
                                                                          name = "attn_dec_train") 
           
        decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, 
                                                                                                                    training_decoder_function,
                                                                                                                    decoder_embedded_input,
                                                                                                                    self.sequence_length,
                                                                                                                    scope = decoding_scope)
        decoder_output_dropout = tf.nn.dropout(decoder_output, self.keep_prob)
        
        return output_function(decoder_output_dropout)
    
    
    def decode_test_set(self, decoder_cell, decoder_embeddings_matrix, decoding_scope, output_function):
        attention_states = tf.zeros([self.params['batch_size'], 1, decoder_cell.output_size])
        
        attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states,
                                                                                                                                        attention_option = 'bahdanau',
                                                                                                                                        num_units = decoder_cell.output_size )
    
        test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function, 
                                                                                  self.encoder_state[0],
                                                                                  attention_keys,
                                                                                  attention_values,
                                                                                  attention_score_function,
                                                                                  attention_construct_function,
                                                                                  decoder_embeddings_matrix, 
                                                                                  self.data.questions_words2int['<SOS>'], 
                                                                                  self.data.questions_words2int['<EOS>'], 
                                                                                  self.sequence_length - 1, 
                                                                                  len(self.data.questions_words2int),
                                                                                  name = "attn_dec_inf") 
        

        test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, 
                                                                                                                    test_decoder_function,
                                                                                                                    scope = decoding_scope)

        return test_predictions         

    def decoder_rnn(self):
        preprocessed_targets = self.preprocess_targets()    
    
        decoder_embeddings_matrix = tf.Variable(tf.random_uniform([len(self.data.questions_words2int) + 1, self.params['decoding_embedding_size']], 0, 1))
    
        decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    
        with tf.variable_scope("decoding") as decoding_scope:
            decoder_cell = self.create_multi_rnn_cell()
            
            weights = tf.truncated_normal_initializer(stddev = 0.1)
            biases = tf.zeros_initializer()
            
            output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                          len(self.data.questions_words2int),
                                                                          scope = decoding_scope,
                                                                          weights_initializer = weights,
                                                                          biases_initializer = biases)
            
            self.training_predictions = self.decode_training_set(decoder_cell,
                                                                 decoder_embedded_input,
                                                                 decoding_scope,
                                                                 output_function)
                
            decoding_scope.reuse_variables()
            
            self.test_predictions = self.decode_test_set(decoder_cell,
                                                         decoder_embeddings_matrix,
                                                         decoding_scope,
                                                         output_function)
        
        
        



































