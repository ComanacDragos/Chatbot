import re

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


# Doing a first cleaning of the texts

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

# Cleaning the questions

clean_questions = [clean_text(x) for x in questions]

# Cleaning the answers

clean_answers = [clean_text(x) for x in answers]

# Filtering out the questions and answers that are too short or too long
short_questions = []
short_answers = []
i = 0
for question in clean_questions:
    if 2 <= len(question.split()) <= 25:
        short_questions.append(question)
        short_answers.append(clean_answers[i])
    i += 1
clean_questions = []
clean_answers = []
i = 0
for answer in short_answers:
    if 2 <= len(answer.split()) <= 25:
        clean_answers.append(answer)
        clean_questions.append(short_questions[i])
    i += 1

# Creating a dictionary that maps each word to its number of occurences

word2count = dict()

for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

# Creating two dictionaries that map the questions words and the answers words to a unique integer

threshold = 20
word_number = 0

questions_words2int = dict()

for word, count in word2count.items():
    if count >= threshold:
        questions_words2int[word] = word_number
        word_number += 1

answers_words2int = dict()

word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        answers_words2int[word] = word_number
        word_number += 1

# Add the last tokens to these two dictionaries

tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']

for token in tokens:
    questions_words2int[token] = len(questions_words2int) + 1

for token in tokens:
    answers_words2int[token] = len(answers_words2int) + 1

# Creating the inverse dictionary of the answers_words2int dictionary

answers_int2word = {word_integer: word for word, word_integer in answers_words2int.items()}

# Adding the End of String token to the end of every answer

for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'

# Translating all the questions and the answers into integers and replacing all
# the words that were filtered out by <OUT>


questions_into_int = list()

for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questions_words2int:
            ints.append(questions_words2int['<OUT>'])
        else:
            ints.append(questions_words2int[word])
    questions_into_int.append(ints)

answers_into_int = list()

for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answers_words2int:
            ints.append(answers_words2int['<OUT>'])
        else:
            ints.append(answers_words2int[word])
    answers_into_int.append(ints)

# Sorting questions and answers by the length of questions

sorted_clean_questions = list()
sorted_clean_answers = list()

for length in range(1, 25 + 1):
    for i in enumerate(questions_into_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])


"""
[15, 230, 36, 25, 197, 65, 66, 8823]
[36, 8824, 8823]
[24, 570, 103, 2411, 8823]


sorted_clean_answers_file = open("Data/sorted_clean_answers.csv", "w")

for answer in sorted_clean_answers:
    string = str(answer[0])
    for nr in answer[1:]:
        string += "," + str(nr)
    print(string)
    sorted_clean_answers_file.write(string + '\n')
"""