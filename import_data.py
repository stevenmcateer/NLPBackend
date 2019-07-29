import matplotlib
import pandas as pd
pd.set_option('display.max_columns', 10)
from sklearn.preprocessing import label_binarize
import time
import torch
import numpy as np
import pandas as pd
import bcolz
import pickle
import re
import sys
from spellchecker import SpellChecker
start_time = time.time()
spell = SpellChecker()
spell.word_frequency.add('scalene')

def load_csv(dataset):
    if dataset == "engage":
        # Open the problems and answers
        engage_ny = pd.read_csv('open_response_filter.csv')

        # Grab correct answers as "y", make numpy array
        y = label_binarize(np.array(engage_ny.correct).astype('str'), classes=['0.0', '0.25', '0.5', '0.75', '1.0'])
        # print("Correct answers x 5")
        # print(y.shape)

        matplotlib.rcParams.update({'errorbar.capsize': 6})

        # Convert to dummy variables
        practice = pd.concat([engage_ny.drop('correct', axis=1),
                              pd.get_dummies(engage_ny['correct'])], axis=1)

        # groupby each problem and sum?
        practice = practice.groupby('problem_id').sum().reset_index()

        # Get number of responses for given problem ID
        check_amount_responses = engage_ny.loc[engage_ny['problem_id'] == 1276708]
        # print("Responses x 1")
        # print(check_amount_responses.shape)


        problems_with_only_grade_1_given = practice.loc[(practice[0.0] == 0) &
                                                        (practice[0.25] == 0) &
                                                        (practice[0.5] == 0) &
                                                        (practice[0.75] == 0)]
        problems_with_only_grade_0_given = practice.loc[(practice[1.0] == 0) &
                                                        (practice[0.25] == 0) &
                                                        (practice[0.5] == 0) &
                                                        (practice[0.75] == 0)]
        dummy_predictors = pd.get_dummies(engage_ny['correct'])
        dummy_predictors.index.values
        return engage_ny, dummy_predictors

    elif dataset == "anthony":
        # Open the problems and answers
        open_responses = pd.read_csv('full_connected_responses.csv', encoding='latin1')
        # print(type(open_responses))
        # print(open_responses)
        return open_responses




def preprocessing(dataset_name):
    spell = SpellChecker()
    spell.word_frequency.add('scalene')

    glove_path = "./glove.6B"

    fully_connected = load_csv("anthony")


    all_problems = fully_connected.groupby('problem_id').count().reset_index()

    # Create word vectors using GLOVE
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.50.dat', mode='w')

    # with open(f'{glove_path}/glove.6B.50d.txt', 'rb') as f:
    #     for l in f:
    #         line = l.decode().split()
    #         word = line[0]
    #         words.append(word)
    #         word2idx[word] = idx
    #         idx += 1
    #         vect = np.array(line[1:]).astype(np.float)
    #         vectors.append(vect)

    vectors = bcolz.carray(vectors[1:].reshape((-1, 50)), rootdir=f'{glove_path}/6B.50.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open(f'{glove_path}/6B.50_words.pkl', 'wb'))
    pickle.dump(word2idx, open(f'{glove_path}/6B.50_idx.pkl', 'wb'))

    vectors = bcolz.open(f'{glove_path}/6B.50.dat')[:]
    words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))

    # Dictionary to get vectors from... glove["the"] = vector(....)
    glove = {w: vectors[word2idx[w]] for w in words}

    master_df = pd.DataFrame(columns=["problem_id", "answer", "grade"])

    # test_prob = [36600, 36601]

    count = 0
    for pid in all_problems["problem_id"]:
        count+=1
        print("Problem: " + str(count) +"/" + str(len(all_problems["problem_id"])) + ": " + str(pid))
        answers, grades = load_X_y(pid, fully_connected, glove)
        # make a problem id list for df
        pid_list = [pid]*len(answers)

        problem_df = pd.DataFrame(
                            {'problem_id': pid_list,
                             'answer': answers,
                             'grade': grades
                            })
        master_df = pd.concat([master_df, problem_df])

    # Save to csv
    master_df.to_csv("vectorized_data.csv", index=False)


def load_X_y(problem_id, dataset, glove):
    problem_object = dataset.loc[dataset['problem_id'] == problem_id]
    answers = []
    cleaned_answers = []
    grades = []

    # Try to clean up answers
    count = 0
    for answer in problem_object.cleaned_answer_text:
        count+=1
        print("Converting answer:", count, "/", len(problem_object.cleaned_answer_text))
        cleaned_answers.append(clean_answer(answer))

    # Find max answer length for matrix
    max_len = 0
    for answer in cleaned_answers:
        num_words = len(answer)
        # print("Is", num_words, "greater than", max_len)
        if num_words > max_len:
            # print("yes, new max is", num_words)
            max_len = num_words

    # print("Max length:", max_len)
    # print(cleaned_answers)

    # Get each answer in order
    # call the function to turn each answer to vectors
    for answer in cleaned_answers:
        # print(answer)
        vector = convert_to_vector(glove, answer, max_len)
        # print(len(vector))
        answers.append(vector)

    # Get each grade in order
    for grade in problem_object.correct:
        # Create one hot encoding for grades
        if grade == 0.0:
            grades.append([1, 0, 0, 0, 0])
        elif grade == 0.25:
            grades.append([0, 1, 0, 0, 0])
        elif grade == 0.50:
            grades.append([0, 0, 1, 0, 0])
        elif grade == 0.75:
            grades.append([0, 0, 0, 1, 0])
        elif grade == 1.0:
            grades.append([0, 0, 0, 0, 1])


    return answers, grades



def clean_answer(sentence):
    sentence_words = list(re.sub("[^\w]", " ",  str(sentence)).split())
    # find those words that may be misspelled
    misspelled = spell.unknown(sentence_words)
    new_sentence = sentence_words
    for word in misspelled:
        # Get the one `most likely` answer
        new_sentence = [w.replace(word,spell.correction(word)) for w in sentence_words]
    return new_sentence



def convert_to_vector(glove, sentence, max_len):
    answer = []
    for word in sentence:
        try:
            answer.append(glove[word])
        except KeyError:
            answer.append(np.random.normal(scale=0.6))

    # Pad with zeroes if needed
    if len(answer) < max_len:
        answer.extend([0] * (max_len - len(answer)))

    # Return vectorized answer
    return answer



if __name__ == '__main__':
    # dataset_name = sys.argv[1]

    preprocessing(dataset_name="anthony")

