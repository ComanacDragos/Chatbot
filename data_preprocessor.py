import re
import json


def movie_corpus_questions_and_answers():
    lines = open("movie_lines.txt", encoding="utf-8", errors="ignore").read().split('\n')

    conversations = open("movie_conversations.txt", encoding="utf-8", errors="ignore").read().split('\n')

    # Creating a dictionary that maps each line_id with its text

    id2line = dict()

    for line in lines:
        split_line = line.split(' +++$+++ ')

        if len(split_line) == 5:
            id2line[split_line[0]] = split_line[-1]

    # Creating a list of lists of conversation ids

    conversations_ids = list()

    for conversation in conversations[:-1]:
        conversation_list = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
        conversations_ids.append(conversation_list.split(','))

    # Get separately the questions and the answers

    questions = list()
    answers = list()

    for conversation in conversations_ids:
        for i in range(len(conversation) - 1):
            questions.append(id2line[conversation[i]])
            answers.append(id2line[conversation[i + 1]])

    return questions, answers


def intents():
    with open("intents.json") as file:
        data = json.load(file)

    questions = list()
    answers = list()

    for intent in data['intents']:
        for pattern in intent['patterns']:
            # for response in intent['responses']:
            questions.append(pattern)
            answers.append(intent['responses'][0])

    return questions, answers


def intents_labels():
    with open("intents.json") as file:
        data = json.load(file)

    questions = list()
    answers = list()

    for intent in data['intents']:
        for pattern in intent['patterns']:
            # for response in intent['responses']:
            questions.append(pattern)
            answers.append(intent['tag'])

    return questions, answers


class DataPreprocessor:
    def __init__(self, questions, answers):
        self.clean_questions = [self.clean_text(q) for q in questions]
        self.clean_answers = [self.clean_text(a) for a in answers]

        # self.filter_questions_and_answers()
        self.create_word2count()
        self.create_words2int()

        self.answers_int2word = {word_integer: word for word, word_integer in self.answers_words2int.items()}

        for i in range(len(self.clean_answers)):
            self.clean_answers[i] += ' <EOS>'

        self.translate_questions_and_answers()
        self.sort_questions_and_answers()

    @staticmethod
    def clean_text(text):
        text = text.lower()
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"what's", "what is", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"how's", "how is", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
        return text

    def filter_questions_and_answers(self):
        # Filtering out the questions and answers that are too short or too long
        short_questions = []
        short_answers = []
        i = 0
        for question in self.clean_questions:
            if 2 <= len(question.split()) <= 25:
                short_questions.append(question)
                short_answers.append(self.clean_answers[i])
            i += 1
        self.clean_questions = []
        self.clean_answers = []
        i = 0
        for answer in short_answers:
            if 2 <= len(question.split()) <= 25:
                self.clean_answers.append(answer)
                self.clean_questions.append(short_questions[i])
            i += 1

    def create_word2count(self):
        self.word2count = dict()

        for question in self.clean_questions:
            for word in question.split():
                if word not in self.word2count:
                    self.word2count[word] = 1
                else:
                    self.word2count[word] += 1

        for answer in self.clean_answers:
            for word in answer.split():
                if word not in self.word2count:
                    self.word2count[word] = 1
                else:
                    self.word2count[word] += 1

    def create_words2int(self):
        threshold = 0
        word_number = 0

        self.questions_words2int = dict()

        for word, count in self.word2count.items():
            if count >= threshold:
                self.questions_words2int[word] = word_number
                word_number += 1

        self.answers_words2int = dict()

        word_number = 0
        for word, count in self.word2count.items():
            if count >= threshold:
                self.answers_words2int[word] = word_number
                word_number += 1

        tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']

        for token in tokens:
            self.questions_words2int[token] = len(self.questions_words2int) + 1

        for token in tokens:
            self.answers_words2int[token] = len(self.answers_words2int) + 1

    def translate_questions_and_answers(self):
        self.questions_into_int = list()

        for question in self.clean_questions:
            ints = []
            for word in question.split():
                if word not in self.questions_words2int:
                    ints.append(self.questions_words2int['<OUT>'])
                else:
                    ints.append(self.questions_words2int[word])
            self.questions_into_int.append(ints)

        self.answers_into_int = list()

        for answer in self.clean_answers:
            ints = []
            for word in answer.split():
                if word not in self.answers_words2int:
                    ints.append(self.answers_words2int['<OUT>'])
                else:
                    ints.append(self.answers_words2int[word])
            self.answers_into_int.append(ints)

    def sort_questions_and_answers(self):
        self.sorted_clean_questions = list()
        self.sorted_clean_answers = list()

        for length in range(1, 25 + 1):
            for i in enumerate(self.questions_into_int):
                if len(i[1]) == length:
                    self.sorted_clean_questions.append(self.questions_into_int[i[0]])
                    self.sorted_clean_answers.append(self.answers_into_int[i[0]])


data = DataPreprocessor(*intents_labels())

with open("hyperparameters.json") as file:
    hyperparams = json.load(file)
