from seq2seq_model import *

checkpoint = "./Weights/chatbot_weights_specific_rnn_v2.ckpt"

session = tf.InteractiveSession()

session.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(session, checkpoint)


# Converting the questions from string to list of encoding integers

def convert_string2int(question, word2int):
    question = data.clean_text(question)
    
    return [word2int.get(word, word2int['<OUT>']) for word in question.split()]


# Setting up the chat
    
while True:
    question = input("You: ")
    
    if question == 'Goodbye':
        break
    
    question = convert_string2int(question, data.questions_words2int)
    
    question = question + [data.questions_words2int['<PAD>']] * (25 - len(question))
    
    fake_batch = np.zeros((batch_size, 25))    
    fake_batch[0] = question
     
    predicted_answer = session.run(test_predictions, {inputs: fake_batch, keep_prob: 0.5})[0]
    
    answer = ''
    
    for i in np.argmax(predicted_answer, 1):
        if data.answers_int2word[i] == 'i':
            token = ' I'
        elif data.answers_int2word[i] == '<EOS>':
            token = '.'
        elif data.answers_int2word[i] == '<OUT>':
            token = ' out'
        else:
            token = ' ' + data.answers_int2word[i]
        
        answer += token
        
        if token == '.':
            break
    
    print('Chatbot: ' + answer)
    


