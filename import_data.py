import matplotlib
import pandas as pd
pd.set_option('display.max_columns', 10)
from sklearn.preprocessing import label_binarize
import nltk
from nltk.corpus import stopwords
import time
import torch
import numpy as np
import pandas as pd
import bcolz
import pickle
import stanfordnlp
from os import path
import ast
start_time = time.time()

#
# stanfordnlp.download('en')   # This downloads the English models for the neural pipeline
# nltk.download("stopwords")
nlp = stanfordnlp.Pipeline(use_gpu=False)

def remake_glove_files():
    glove_path = "./glove.6B"
    # Create word vectors using GLOVE
    words = []
    idx = 0
    word2idx = {}
    glove = {}
    vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.100.dat', mode='w')

    with open(f'{glove_path}/glove.6B.100d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)


    vectors = bcolz.carray(vectors[1:].reshape((-1, 100)), rootdir=f'{glove_path}/6B.100.dat', mode='w')
    vectors.flush()

    # Save everything
    with open("./glove.6B/6B.100_words.pkl", "wb") as f:
        pickle.dump(words, f)

    with open("./glove.6B/6B.100_idx.pkl", "wb") as g:
        pickle.dump(word2idx, g)

    # Dictionary to get vectors from... glove["the"] = vector(....)
    glove = {w: vectors[word2idx[w]] for w in words}

    with open("./glove.6B/6B.100_glove.pkl", "wb") as g:
        pickle.dump(glove, g)

    return words, word2idx, glove



def open_saved_glove_files():
    # Open saved stuff
    vectors = bcolz.open('./glove.6B/6B.100.dat')[:]
    try:
        with open('./glove.6B/6B.100_words.pkl', 'rb') as f:
            words_data = pickle.load(f)
            # print(words_data)

        with open('./glove.6B/6B.100_idx.pkl', 'rb') as f:
            idx_data = pickle.load(f)
            # print(idx_data)

        with open('./glove.6B/6B.100_glove.pkl', 'rb') as f:
            glove = pickle.load(f)


    except FileNotFoundError:
        print("Sorry")

    return words_data, idx_data, glove

def get_dataset_vocab(master_df, dataset):
    if dataset == "glove1":
        try:
            with open('./glove.6B/vocab_list_pad.pkl', 'rb') as f:
                vocab_list = pickle.load(f)
                return vocab_list
        except FileNotFoundError:
            vocab_list = []
            vocab_list.append("<pad>")
            for answer in master_df.cleaned_answer_text:

                for word in clean_answer(answer):
                    if word not in vocab_list:
                        vocab_list.append(word)

                        # print(vocab_list)
            with open("./glove.6B/vocab_list_pad.pkl", "wb") as f:
                pickle.dump(vocab_list, f)
    elif dataset == "glove2":
        try:
            with open('./glove.6B/vocab_list_glove2.pkl', 'rb') as f:
                vocab_list = pickle.load(f)
                return vocab_list
        except FileNotFoundError:
            vocab_list = []
            vocab_list.append("<pad>")
            count = 1
            for answer in master_df.parsed_cleaned_answers2:
                count+=1
                print("Row", count, "of", len(master_df.parsed_cleaned_answers2))
                print(answer)
                for word in answer:
                    print("word", word)
                    if word not in vocab_list:
                        vocab_list.append(word)

                        # print(vocab_list)
            with open("./glove.6B/vocab_list_glove2.pkl", "wb") as f:
                pickle.dump(vocab_list, f)
    elif dataset == "glove3":
        try:
            with open('./glove.6B/vocab_list_glove3.pkl', 'rb') as f:
                vocab_list = pickle.load(f)
                return vocab_list
        except FileNotFoundError:
            vocab_list = []
            vocab_list.append("<pad>")
            count = 1
            for answer in master_df.parsed_cleaned_answers2:
                count += 1
                print("Row", count, "of", len(master_df.parsed_cleaned_answers2))
                print(answer)
                for word in answer:
                    print("word", word)
                    if word not in vocab_list:
                        vocab_list.append(word)

                        # print(vocab_list)
            with open("./glove.6B/vocab_list_glove3.pkl", "wb") as f:
                pickle.dump(vocab_list, f)

    print("length of final vocab list", len(vocab_list))
    return vocab_list



def create_weights_matrix(dataset_vocab_list, glove, dataset_name):
    try:
        with open('./glove.6B/weights_matrix_100d_' + dataset_name + '.pkl', 'rb') as f:
            weights_matrix = pickle.load(f)
            weights_matrix = torch.tensor(weights_matrix, dtype=torch.float)

        with open('./glove.6B/dataset_w2idx_100d_' + dataset_name + '.pkl', 'rb') as f:
            dataset_word2idx = pickle.load(f)

        return weights_matrix, dataset_word2idx
    except FileNotFoundError:
        matrix_len = len(dataset_vocab_list)
        # dim is 100 because of glove.6B100
        weights_matrix = np.zeros((matrix_len, 100))
        dataset_word2idx = {}
        words_found = 0

        for i, word in enumerate(dataset_vocab_list):
            # if the pad vocab word
            if i == 0:
                print("pad word at index 0 is", word)
            dataset_word2idx[word] = i

            print("Word", i, "of", matrix_len)
            try:
                weights_matrix[i] = glove[word]
                words_found += 1
                print("Found words:", words_found)
            except KeyError:
                weights_matrix[i] = np.random.normal(scale=0.6, size=(100,))
            print("index", i, "has vector", weights_matrix[i])

        with open("./glove.6B/weights_matrix_100d_" + dataset_name +".pkl", "wb") as f:
            pickle.dump(weights_matrix, f)


        with open("./glove.6B/dataset_w2idx_100d_" + dataset_name + ".pkl", "wb") as f:
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

    elif dataset == "glove1":
        # Open the problems and answers
        open_responses = pd.read_csv('full_connected_responses.csv', encoding='latin1')
        return open_responses
    elif dataset == "glove2":
        # Open the problems and answers
        open_responses = pd.read_csv('tokenized_glove2_lower.csv', converters={6: ast.literal_eval}, encoding='latin1')
        return open_responses
    elif dataset == "glove3":
        # Open the problems and answers
        open_responses = pd.read_csv('tokenized_glove3.csv', converters={6: ast.literal_eval}, encoding='latin1')
        return open_responses


def preprocessing(dataset_name):
    file_name = "vectorized_" + dataset_name + ".csv"
    if path.exists(file_name):
        master_df = pd.read_csv(file_name, converters={4: ast.literal_eval, 6: ast.literal_eval})
        # master_df = master_df.astype({'answer': 'list', 'grade': 'list'}, copy=True)

        # if 'grader_teacher_id' not in master_df.columns:
        #     fully_connected = load_csv(dataset_name)
        #     cleaned_columns_connected = fully_connected[
        #         ["grader_teacher_id", "problem_log_id", "problem_id", "cleaned_answer_text", "grade", "folds"]]
        #     # Add the grader teach column to vectorized dataset
        #     master_df['grader_teacher_id'] = cleaned_columns_connected['grader_teacher_id'].astype(int)
        #
        #
        #     # for index, row in cleaned_columns_connected.iterrows():
        #     #     if pd.isnull(row['cleaned_answer_text']):
        #     #         master_df = master_df.drop(index=index)
        #
        #     # master_df.reset_index(inplace=True)
        #     all_problems = master_df.groupby('problem_id').count().reset_index()
        #     master_df.to_csv("vectorized_anthony_raw_length_pad_idx.csv_teacher.csv")
        print(master_df)
        all_problems = master_df.groupby('problem_id').count().reset_index()
        # Load saved pickled files
        words_data, word2idx, glove = open_saved_glove_files()
        vocab = get_dataset_vocab(master_df, dataset_name)
        weights_matrix, dataset_word2idx = create_weights_matrix(vocab, glove, dataset_name)
        return master_df, all_problems, vocab, weights_matrix
    else:

        fully_connected = load_csv(dataset_name)

        cleaned_columns_connected = fully_connected[["grader_teacher_id", "problem_log_id", "problem_id", "cleaned_answer_text", "grade","folds"]]
        # cleaned_columns_connected = fully_connected[
        #     ["problem_id", "parsed_cleaned_answers", "grade", "folds"]]

        for index, row in cleaned_columns_connected.iterrows():
            if pd.isnull(row['cleaned_answer_text']):
                print("Removing null answer")
                cleaned_columns_connected = cleaned_columns_connected.drop(index=index)


        all_problems = cleaned_columns_connected.groupby('problem_id').count().reset_index()

        # Load saved
        words_data, word2idx, glove = open_saved_glove_files()
        # # Remake saved pickled files
        # words_data, word2idx, glove = remake_glove_files()
        vocab = get_dataset_vocab(cleaned_columns_connected, dataset_name)
        weights_matrix, dataset_word2idx = create_weights_matrix(vocab, glove, dataset_name)

        # master_df = pd.DataFrame(columns=["grader_teacher_id", "problem_log_id", "problem_id", "answer", "grade", "folds"])
        master_df = pd.DataFrame(
            columns=["problem_id", "answer", "grade", "folds"])


        # test_prob = [36600, 36601]

        count = 0
        for index, row in cleaned_columns_connected.iterrows():
        # index = 23177
        # row = cleaned_columns_connected.iloc[23177]
            count+=1
            # cleaned_columns_connected.loc[:, 'grade'] = cleaned_columns_connected['grade'].astype(str)
            pid = row["problem_id"]
            answer = row["cleaned_answer_text"]
            problem_log_id = row["problem_log_id"]
            folds = row["folds"]
            grader_teacher_id = row["grader_teacher_id"]
            grade = row["grade"]

            print("Row: " + str(count) +"/" + str(len(cleaned_columns_connected)) + ": " + str(pid))
            # Convert answer to vector
            cleaned_answer = clean_answer(answer)
            idx_answer = convert_to_idx_vector(dataset_word2idx, cleaned_answer)

            # Convert grade to one hot
            encoded_grade = convert_grade(grade)

            master_df = master_df.append({'problem_log_id': problem_log_id, 'problem_id': pid, 'answer':str(cleaned_answer), 'idx_answer':str(idx_answer), 'grade':str(grade), 'encoded_grade':str(encoded_grade), 'folds': folds, "grader_teacher_id": grader_teacher_id},  ignore_index=True)
            # master_df = master_df.append({'problem_id': pid, 'answer':str(idx_answer), 'grade':str(encoded_grade), 'folds': folds},  ignore_index=True)
        # print(master_df)

        # Save to csv
        master_df.to_csv("tokenized_" + dataset_name + ".csv", index=False)


        return master_df, all_problems, vocab, weights_matrix


def load_X_y(problem_id, dataset, dataset_name, glove):
    problem_object = dataset.loc[dataset['problem_id'] == problem_id]
    answers = []
    cleaned_answers = []
    grades = []

    # Try to clean up answers
    count = 0
    if dataset_name=="glove1":
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

def clean_answer(sentence):
    # use stanford tokenizer
    stopWords = set(stopwords.words('english'))
    print("Sentence is:" + "'" + str(sentence) + "'")
    sentence = str(sentence).lower()

    # remove bad_chars
    sentence = sentence.replace('\n', '').replace('\r', '').replace('\r\n', '')

    sentence_words = []
    if len(sentence) > 0:
        # print("length is greater than 0")
        doc = nlp(sentence)
        for i, sent in enumerate(doc.sentences):
            for word in sent.words:
                # if word.text not in stopWords:
                    # print(word, " not in", stopWords)
                sentence_words.append(word.text)

            # for word in sentence_trimmed:
            #     sentence_words.append(word.text)
        print("New sentence is", sentence_words)
    else:
        # print("Length is 0")
        return []
    return sentence_words




def convert_to_idx_vector(dataset_word2idx, sentence):
    answer = []
    # print(len(dataset_word2idx))
    count = 0
    for word in sentence:
        # try:
        # print(dataset_word2idx)
        # print("word is", word, "index is", dataset_word2idx[word])
        try:
            index = dataset_word2idx[word]
        except KeyError:
            count += 1
            index = 0

        answer.append(index)

    print(count, "words not found in dataset2idx")

    # Return vectorized answer
    return answer

