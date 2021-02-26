# Chatbot

Requirements: 
- Tensorflow 1.0.0
- Python 3.5
- Cornell movie dialogs corpus

This is a learning project in which I experimented with recurent neural networks and long-short term memory cells.

The chatbot is inspired from the course “Udemy-Deep Learning and NLP A-Z How to create a Chatbot”.

The project is composed of 4 main parts

1. Data preprocessor - process the movie lines, that includes cleaning the words and converting them to numbers
2. Seq2Seq model - the implementation of the recurent neural network, using LSTM's (this model is suitable for processing sequences of objects, such as words in this case)
3. Training - separating training data and validation data; and then actually training the model on the training data and the testing the accuracy on validation data (based on this, the weights are updated)
4. UI - simple text interface for training and chatting

The different parameters can be set from hyperparameters.json which includes the following:
- epochs: the number of times the whole data set is processed
- batch_size: the number of sequences that are being fed into the model at a time (this is usefull for stochastic gradient descent meaning that the weights are updated after a batch is processed)
- rnn_size: number of neurons
- num_layers: number of layers
- encoding_embedding_size: the size of the encoding matrix
- decoding_embedding_size: the size of the decoding matrix
- learning_rate: the rate at which the weights are updated (the rate at which the gradient approaches a minimum)
- learning_rate_decay: the rate at which the learning rate is decreased (this helps the gradient descent such that it reaches a minimum)
- min_learning_rate: the minimum learning rate
- keep_probability: the probability that a neuron is active during training/chatting (this helps reduce overfiting)
- seq_length: the length of a sequence (the model needs fixed length sequences, so for example a proposition of 10 words is padded with 15 <PAD> elements to form a sequence)
- training_validation_split: the procent of data that is reserved for validation
- batch_index_check_training_loss: after this number of batches information will be printed
- early_stopping_stop: if there is no improvement after this number of batches the training is halted
- checkpoint: file where the weights are saved
- print_batch_data: true if data is to be printed after each batch, false otherwise


![Second chat](https://user-images.githubusercontent.com/46956225/109343353-e5fcf880-7875-11eb-92aa-dd164e1a522d.png)
<br>One of the firsts chats :)
