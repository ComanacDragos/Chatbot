from seq2seq_model import *
import time
import numpy as np
import tensorflow as tf


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

    def apply_padding(self, batch_of_sequences):
        """
        Adauga simbolul <PAD> la finalul tuturor intrebarilor si raspunsurilor pentru a avea aceeasi lungime
        """
        max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])

        return [sequence + [self.model.data.questions_words2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]

    def split_into_batches(self, questions, answers):
        """
        Imparte intrebarile si raspunsurile in grupe de o anumita marime
        """
        for batch_index in range(len(questions) // self.model.params['batch_size']):
            start_index = batch_index * self.model.params['batch_size']
            questions_in_batch = questions[start_index : start_index + self.model.params['batch_size']]
            answers_in_batch = answers[start_index : start_index + self.model.params['batch_size']]
            
            padded_questions_in_batch = np.array(self.apply_padding(questions_in_batch))
            padded_answers_in_batch = np.array(self.apply_padding(answers_in_batch))
    
            yield padded_questions_in_batch, padded_answers_in_batch

    def store_weights(self, file):
        saver = tf.train.Saver()
        saver.save(self.session, file)
        print("Weights saved")
        
    def load_weights(self, file):
        saver = tf.train.Saver()
        saver.restore(self.session, file)

    def start_train_loop(self):
        """
        Incepe bucla principala pe epoci
        print_batch_data - variabila booleana : printeaza datele pentru fiecare batch daca este adevarat
        """
            
        batch_index_check_validation_loss = ((len(self.training_questions)) // self.model.params['batch_size'] // 2) - 1

        total_training_loss_error = 0
        list_validation_loss_error = []
        
        early_stopping_check = 0
        
        learning_rate = self.model.params['learning_rate']
        
        self.session.run(tf.global_variables_initializer())

        print("Starting training")
        for epoch in range(1, self.model.params['epochs'] + 1):
            for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(self.split_into_batches(self.training_questions, self.training_answers)):
                starting_time = time.time()
                
                _, batch_training_loss_error = self.session.run([self.optimizer_gradient_clipping, self.loss_error], {self.model.inputs : padded_questions_in_batch,
                                                                                                                 self.model.targets : padded_answers_in_batch,
                                                                                                                 self.model.learning_rate: learning_rate,
                                                                                                                 self.model.sequence_length : padded_answers_in_batch.shape[1],
                                                                                                                 self.model.keep_prob : self.model.params['keep_probability']})
                ending_time = time.time()
                batch_time = ending_time - starting_time
                
                if not self.model.params['print_batch_data']:
                    total_training_loss_error += batch_training_loss_error
                                
                    if batch_index % self.model.params['batch_index_check_training_loss'] == 0:
                        print("Epoch : {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.6f}, Training Time on {:d} batches: {:d} seconds".format(epoch,
                                                                                                                                                    self.model.params['epochs'],
                                                                                                                                                    batch_index,
                                                                                                                                                    len(self.training_questions) // self.model.params['batch_size'],
                                                                                                                                                    total_training_loss_error / self.model.params['batch_index_check_training_loss'],
                                                                                                                                                    self.model.params['batch_size'],
                                                                                                                                                    int(batch_time * self.model.params['batch_index_check_training_loss'])))
                        total_training_loss_error = 0
                else:
                    print("Batch : {:d} Batch time : {:d} seconds Batch training loss error: {:>6.6f}".format(batch_index,
                                                                                                                          int(batch_time),
                                                                                                                          batch_training_loss_error))
                    
                if batch_index % batch_index_check_validation_loss: 
                    total_validation_loss_error = 0
                    
                    starting_time = time.time()
                    
                    for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(self.split_into_batches(self.validation_questions, self.validation_answers)):
                        batch_validation_loss_error = self.session.run(self.loss_error, {self.model.inputs: padded_questions_in_batch,
                                                                                         self.model.targets: padded_answers_in_batch,
                                                                                         self.model.learning_rate: learning_rate,
                                                                                         self.model.sequence_length: padded_answers_in_batch.shape[1],
                                                                                         self.model.keep_prob: 1})
                        if self.model.params['print_batch_data']:
                            print("Validation batch : {:d}, Batch training loss error: {:>6.6f}".format(batch_index_validation, batch_validation_loss_error))
                        total_validation_loss_error += batch_validation_loss_error
                    
                    ending_time = time.time()     
                    batch_time = ending_time - starting_time   
                    
                    average_validation_loss_error = total_validation_loss_error / (len(self.validation_questions) / self.model.params['batch_size'])
                
                    print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))

                    learning_rate *= self.model.params['learning_rate_decay']
                    
                    if learning_rate < self.model.params['min_learning_rate']:
                        learning_rate = self.model.params['min_learning_rate']
                        
                    list_validation_loss_error.append(average_validation_loss_error)
                    
                    if average_validation_loss_error <= min(list_validation_loss_error):
                        print("I speak better now")
                        early_stopping_check = 0
                        self.store_weights(self.model.params['checkpoint'])
                    else:
                        print("Sorry I do not speak better")
                        early_stopping_check += 1
                        
                        if early_stopping_check == self.model.params['early_stopping_stop']:
                            break
            if early_stopping_check == self.model.params['early_stopping_stop']:
                print("I can't speak better anymore")
                break        
    


        print("Finished training")
                                


























