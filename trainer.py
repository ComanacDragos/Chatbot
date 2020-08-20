from seq2seq_model import *
import time
import numpy as np


class Trainer:
    def __init__(self, model):        
        self.session = tf.InteractiveSession()

        self.model = model
        
        #Dimensiunea tensorului de inputuri
        self.input_shape = tf.shape(self.model.inputs)
        
        self.training_predictions, self.test_predictions = self.model.get_predictions()

                
        with tf.name_scope("optimization"):
            self.loss_error = tf.contrib.seq2seq.sequence_loss(self.training_predictions,
                                                               self.model.targets,
                                                               tf.ones([self.input_shape[0], self.model.sequence_length]))
            
            optimizer = tf.train.AdamOptimizer(self.model.params['learning_rate'])
            gradients = optimizer.compute_gradients(self.loss_error)
            clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
            
            self.optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)
            
        self.setup_data()
        
    def setup_data(self):
        """
        Impart datele in cele pentru antrenare si cele pentru validare
        """
        training_validation_split = int(len(self.model.data.sorted_clean_questions) * self.model.params['training_validation_split'])
        
        self.training_questions = self.model.data.sorted_clean_questions[training_validation_split:]
        self.validation_questions = self.model.data.sorted_clean_questions[:training_validation_split]
        
        self.training_answers = self.model.data.sorted_clean_answers[training_validation_split:]
        self.validation_answers = self.model.data.sorted_clean_answers[:training_validation_split]

        def apply_padding(batch_of_sequences):
            """
            Adauga simbolul <PAD> la finalul tuturor intrebarilor si raspunsurilor pentru a avea aceeasi lungime
            """
            max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    
            return [sequence + [self.model.data.questions_words2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]
    
        def split_into_batches(questions, answers):
            """
            Imparte intrebarile si raspunsurile in grupe de o anumita marime
            """
            for batch_index in range(len(questions) // self.model.params['batch_size']):
                start_index = batch_index * self.model.params['batch_size']
                questions_in_batch = questions[start_index : start_index + self.model.params['batch_size']]
                answers_in_batch = answers[start_index : start_index + self.model.params['batch_size']]
                
                padded_questions_in_batch = np.array(apply_padding(questions_in_batch, self.model.data.questions_words2int))
                padded_answers_in_batch = np.array(apply_padding(answers_in_batch, self.model.data.answers_words2int))
        
                yield padded_questions_in_batch, padded_answers_in_batch

        def start_train_loop(self, print_batch_data):
            """
            Incepe bucla principala pe epoci
            print_batch_data - variabila booleana : printeaza datele pentru fiecare batch daca este adevarat
            """
                
            batch_index_check_validation_loss = ((len(self.training_questions)) // self.model.params['batch_size'] // 2) - 1
    
            total_training_loss_error = 0
            list_validation_loss_error = []
            
            early_stopping_check = 0   
            
            session.run(tf.global_variables_initializer())

            for epoch in range(1, self.model.params['epochs'] + 1):
                for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(self.training_questions, self.training_answers, self.model.params['batch_size'])):
                    starting_time = time.time()
                    
                    
                    first_output, batch_training_loss_error = self.session.run([self.optimizer_gradient_clipping, self.loss_error], {self.model.inputs : padded_questions_in_batch,
                                                                                                                     self.model.targets : padded_answers_in_batch,
                                                                                                                     self.model.learning_rate: self.model.params['learning_rate'],
                                                                                                                     self.model.sequence_length : padded_answers_in_batch.shape[1],
                                                                                                                     self.model.keep_prob : self.model.params['keep_probability']})
                    ending_time = time.time()
                    batch_time = ending_time - starting_time
                    
                    if not print_batch_data:
                        total_training_loss_error += batch_training_loss_error
                                    
                        if batch_index % self.model.params['batch_index_check_training_loss'] == 0:
                            print('Epoch : {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.6f}, Training Time on {:d} batches: {:d} seconds'.format(epoch,
                                                                                                                                                        self.model.params['epochs'],
                                                                                                                                                        batch_index,
                                                                                                                                                        len(self.training_questions) // self.model.params['batch_size'],
                                                                                                                                                        total_training_loss_error / self.model.params['batch_index_check_training_loss'],
                                                                                                                                                        self.model.params['batch_size'],
                                                                                                                                                        int(batch_time * self.model.params['batch_index_check_training_loss'])))
                            total_training_loss_error = 0
                    else:
                        print("Batch : {:d} Batch time : {:d} seconds Batch training loss error: {:>6.6f}, FirstOutput: {:d}".format(batch_index,
                                                                                                                              int(batch_time),
                                                                                                                              batch_training_loss_error,
                                                                                                                              first_output))
                    
tf.reset_default_graph()
        
model = Seq2SeqModel(data, hyperparams)
trainer = Trainer(model)





























