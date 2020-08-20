from seq2seq_model import *


class Trainer:
    def __init__(self, model):        
        self.session = tf.InteractiveSession()

        self.model = model
        
        #Dimensiunea tensorului de inputuri
        self.input_shape = tf.shape(self.model.inputs)
        
        self.training_predictions, self.test_predictions = self.model.get_predictions()

        

data = DataPreprocessor(*intents_labels())

with open("hyperparameters.json") as file:
    hyperparams = json.load(file)


tf.reset_default_graph()
        
model = Seq2SeqModel(data, hyperparams)
trainer = Trainer(model)
