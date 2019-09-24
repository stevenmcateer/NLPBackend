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
import stanfordnlp
import re
import sys
from spellchecker import SpellChecker
import os.path
from os import path
import ast
import math
start_time = time.time()


# stanfordnlp.download('en')   # This downloads the English models for the neural pipeline


def remake_glove_files():
    glove_path = "./glove.6B"
    # Create word vectors using GLOVE
    words = []
    idx = 0
    word2idx = {}
    glove = {}
    vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.50.dat', mode='w')

    with open(f'{glove_path}/glove.6B.50d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)


    vectors = bcolz.carray(vectors[1:].reshape((-1, 100)), rootdir=f'{glove_path}/6B.50d.dat', mode='w')
    vectors.flush()

    # Save everything
    with open("./glove.6B/6B.50_words.pkl", "wb") as f:
        pickle.dump(words, f)

    with open("./glove.6B/6B.50_idx.pkl", "wb") as g:
        pickle.dump(word2idx, g)

    # Dictionary to get vectors from... glove["the"] = vector(....)
    glove = {w: vectors[word2idx[w]] for w in words}

    with open("./glove.6B/6B.50_glove.pkl", "wb") as g:
        pickle.dump(glove, g)

    return words, word2idx, glove



def open_saved_glove_files():
    # Open saved stuff
    vectors = bcolz.open('./glove.6B/6B.50.dat')[:]
    try:
        with open('./glove.6B/6B.50_words.pkl', 'rb') as f:
            words_data = pickle.load(f)
            # print(words_data)

        with open('./glove.6B/6B.50_idx.pkl', 'rb') as f:
            idx_data = pickle.load(f)
            # print(idx_data)

        with open('./glove.6B/6B.50_glove.pkl', 'rb') as f:
            glove = pickle.load(f)


    except FileNotFoundError:
        print("Sorry")

    return words_data, idx_data, glove

def get_dataset_vocab(master_df):
    try:
        with open('./glove.6B/vocab_list.pkl', 'rb') as f:
            vocab_list = pickle.load(f)
            return vocab_list
    except FileNotFoundError:
        vocab_list = []
        for answer in master_df.cleaned_answer_text:

            for word in clean_answer(answer):
                if word not in vocab_list:
                    vocab_list.append(word)

                    # print(vocab_list)
        with open("./glove.6B/vocab_list.pkl", "wb") as f:
            pickle.dump(vocab_list, f)

        print("final", vocab_list)
        return vocab_list



def create_weights_matrix(dataset_vocab_list, glove):

    try:
        with open('./glove.6B/weights_matrix.pkl', 'rb') as f:
            weights_matrix = pickle.load(f)
            weights_matrix = torch.tensor(weights_matrix, dtype=torch.float)

        with open('./glove.6B/dataset_w2idx.pkl', 'rb') as f:
            dataset_word2idx = pickle.load(f)

        return weights_matrix, dataset_word2idx
    except FileNotFoundError:
        matrix_len = len(dataset_vocab_list)
        # dim is 50 because of glove.6B50
        weights_matrix = np.zeros((matrix_len, 50))
        dataset_word2idx = {}
        words_found = 0

        for i, word in enumerate(dataset_vocab_list):
            dataset_word2idx[word] = i

            print("Word", i, "of", matrix_len)
            try:
                weights_matrix[i] = glove[word]
                words_found += 1
                print("Found words:", words_found)
            except KeyError:
                weights_matrix[i] = np.random.normal(scale=0.6, size=(50,))

        with open("./glove.6B/weights_matrix.pkl", "wb") as f:
            pickle.dump(weights_matrix, f)


        with open("./glove.6B/dataset_w2idx.pkl", "wb") as f:
            pickle.dump(dataset_word2idx, f)
    return weights_matrix, dataset_word2idx

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
        return engage_ny #, dummy_predictors

    elif dataset == "anthony":
        # Open the problems and answers
        open_responses = pd.read_csv('full_connected_responses.csv', encoding='latin1')
        return open_responses


def preprocessing(dataset_name):


    if path.exists("vectorized_" + dataset_name + "_folds_tokenize.csv"):
        master_df = pd.read_csv("vectorized_" + dataset_name + "_folds_tokenize.csv", converters={2: ast.literal_eval, 3: ast.literal_eval})
        all_problems = master_df.groupby('problem_id').count().reset_index()
        # Load saved pickled files
        words_data, word2idx, glove = open_saved_glove_files()
        vocab = get_dataset_vocab(master_df)
        weights_matrix, dataset_word2idx = create_weights_matrix(vocab, glove)
        return master_df, all_problems, vocab, weights_matrix
    else:
        nlp = stanfordnlp.Pipeline()
        fully_connected = load_csv(dataset_name)
        cleaned_columns_connected = fully_connected[["problem_log_id", "problem_id", "cleaned_answer_text", "grade", "folds"]]
        # print(cleaned_columns_connected)


        all_problems = cleaned_columns_connected.groupby('problem_id').count().reset_index()

        # Load saved
        words_data, word2idx, glove = remake_glove_files()
        # # Remake saved pickled files
        # words_data, word2idx, glove = remake_glove_files()
        vocab = get_dataset_vocab(cleaned_columns_connected)
        weights_matrix, dataset_word2idx = create_weights_matrix(vocab, glove)



        master_df = pd.DataFrame(columns=["problem_log_id", "problem_id", "answer", "grade", "folds"])

        # test_prob = [36600, 36601]

        count = 0
        for index, row in cleaned_columns_connected.iterrows():
        # index = 23177
        # row = cleaned_columns_connected.iloc[23177]
            count+=1
            cleaned_columns_connected.loc[:, 'grade'] = cleaned_columns_connected['grade'].astype(str)
            pid = row["problem_id"]
            answer = row["cleaned_answer_text"]
            problem_log_id = row["problem_log_id"]
            folds = row["folds"]
            if pd.isnull(answer):
                answer = ""
            grade = row["grade"]
            # Get the max answer length
            # max_response_len = get_max_response_length(pid, cleaned_columns_connected)
            max_response_len = 20
            print(max_response_len)

            print("Row: " + str(count) +"/" + str(len(cleaned_columns_connected)) + ": " + str(pid))
            # Convert answer to vector
            cleaned_answer = clean_answer(nlp, answer)
            idx_answer = convert_to_idx_vector(dataset_word2idx, cleaned_answer, max_response_len)
            # Convert grade to one hot
            encoded_grade = convert_grade(grade)
            cleaned_columns_connected.at[index, 'cleaned_answer_text'] = str(idx_answer)
            cleaned_columns_connected.at[index, 'grade'] = str(encoded_grade)

            master_df = master_df.append({'problem_log_id': problem_log_id, 'problem_id': pid, 'answer':str(idx_answer), 'grade':str(encoded_grade), 'folds': folds},  ignore_index=True)
            print(master_df)

        # Save to csv
        master_df.to_csv("vectorized_" + dataset_name + "_stanford_100d.csv", index=False)


        return master_df, all_problems, vocab, weights_matrix, dataset_word2idx

def get_max_response_length(pid, df):
    problem_object = df.loc[df['problem_id'] == pid]
    # Find max answer length for matrix
    max_len = 0
    for response in problem_object.cleaned_answer_text:
        num_words = len(clean_answer(response))

        # print("Is", num_words, "greater than", max_len)
        if num_words > max_len:
            # print("yes, new max is", num_words)
            max_len = num_words

    return max_len

def load_X_y(problem_id, dataset, dataset_name, glove):
    problem_object = dataset.loc[dataset['problem_id'] == problem_id]
    answers = []
    cleaned_answers = []
    grades = []

    # Try to clean up answers
    count = 0
    if dataset_name=="anthony":
        for answer in problem_object.cleaned_answer_text:
            count+=1
            print("Converting answer:", count, "/", len(problem_object.cleaned_answer_text))
            cleaned_answers.append(clean_answer(answer))
    elif dataset_name=="engage":
        for answer in problem_object.answer_text:
            count+=1
            print("Converting answer:", count, "/", len(problem_object.answer_text))
            cleaned_answers.append(clean_answer(answer))



def convert_grade(grade):
    grade = str(grade)
    # Create one hot encoding for grades
    if grade == "1":
        return [1, 0, 0, 0, 0]
    elif grade == "2":
        return [0, 1, 0, 0, 0]
    elif grade == "3":
        return [0, 0, 1, 0, 0]
    elif grade == "4":
        return [0, 0, 0, 1, 0]
    elif grade == "5":
        return [0, 0, 0, 0, 1]
    return [0, 0, 0, 0, 0]

def clean_answer(nlp, sentence):
    # use stanford tokenizer
    print("Sentence is:" + "'" + sentence + "'")
    sentence = sentence.lower()

    # CORRECT_SENTENCE_LENGTH
    # AMBIGUOUS_SENTENCE_LENGTH

    # for c in sentence:
    #     print(ord(c))

    # using join() + generator to
    # remove bad_chars
    sentence = sentence.replace('\n', '').replace('\r', '').replace('\r\n', '')
    # print(sentence)
    #
    # for c in sentence:
    #     print(ord(c))
    sentence_words = []
    if len(sentence) > 0:
        # print("length is greater than 0")
        doc = nlp(sentence)
        for i, sent in enumerate(doc.sentences):
            sentence_trimmed = sent.words[:, 20]
            for word in sentence_trimmed:
                sentence_words.append(word.text)
        print(sentence_words)
    else:
        # print("Length is 0")
        return []

    # sentence = str(sentence)
    # # initializing bad_chars_list
    # bad_chars = [';', ':', '!', '\n', '\r']
    #
    # # using join() + generator to
    # # remove bad_chars
    # cleaned = sentence
    # for i in bad_chars:
    #     cleaned = sentence.replace(i, '')
    #
    # sentence_words = list(re.sub("[^\w]", " ", str(cleaned)).split())
    # print(sentence_words)

    return sentence_words

    # # find those words that may be misspelled
    # misspelled = spell.unknown(sentence_words)
    # new_sentence = sentence_words
    # count = 0
    # for word in misspelled:
    #     print(count)
    #     count+=1
    #     if count > 10:
    #         return new_sentence
    #     else:
    #         # Get the one `most likely` answer
    #         print("Replacing", word, "with", spell.correction(word))
    #         new_sentence = [w.replace(word,spell.correction(word)) for w in sentence_words]
    #         print(new_sentence)
    # return new_sentence



def convert_to_idx_vector(dataset_word2idx, sentence, max_len):
    answer = []
    # print(len(dataset_word2idx))

    for word in sentence:
        # try:
        # print(dataset_word2idx)
        # print("word is", word, "index is", dataset_word2idx[word])
        try:
            index = dataset_word2idx[word]
        except KeyError:
            index = 0

        answer.append(index)

    # Pad with zeroes if needed
    if len(answer) < max_len:
        answer.extend([0] * (max_len - len(answer)))

    # Return vectorized answer
    return answer


#
if __name__ == '__main__':
    # dataset_name = sys.argv[1]

    preprocessing(dataset_name="anthony")

