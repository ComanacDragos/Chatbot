from trainer import *
import tensorflow as tf

class UserInterface:
    def __init__(self, data, hyperparams):
        self.data = data
        self.params = hyperparams
        
        tf.reset_default_graph() 
        
        self.model = Seq2SeqModel(self.data, self.params)
        self.trainer = Trainer(self.model)
        
        self.session = self.trainer.session
        
        print("""
Choose 1 / 2
1. Train
2. Chat
""")
        choice = int(input(">> "))
        
        if choice == 1:
            self.train()
        else:
            self.chat()
    
    def train(self):        
        self.trainer.start_train_loop()
    
    def convert_string2int(self, question):
        question = self.data.clean_text(question)
    
        return [self.data.questions_words2int.get(word, self.data.questions_words2int['<OUT>']) for word in question.split()]

    
    def chat(self):
        self.session.run(tf.global_variables_initializer())
        
        saver = tf.train.Saver()
        saver.restore(self.session, self.params['checkpoint'])

        while True:
            question = input("You: ")
            
            if question == 'Goodbye':
                break
            
            question = self.convert_string2int(question)
            
            question = question + [self.data.questions_words2int['<PAD>']] * (25 - len(question))
            
            fake_batch = np.zeros((self.params['batch_size'], 25))    
            fake_batch[0] = question
             
            predicted_answer = self.session.run(self.model.test_predictions, {self.model.inputs: fake_batch, self.model.keep_prob: 0.5})[0]
            
            answer = ''
            
            for i in np.argmax(predicted_answer, 1):
                if self.data.answers_int2word[i] == 'i':
                    token = ' I'
                elif self.data.answers_int2word[i] == '<EOS>':
                    token = '.'
                elif self.data.answers_int2word[i] == '<OUT>':
                    token = ' out'
                else:
                    token = ' ' + self.data.answers_int2word[i]
                
                answer += token
                
                if token == '.':
                    break
            
            print('Chatbot: ' + answer)
            
                
        
UserInterface(data, hyperparams)

        
        
