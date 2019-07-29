import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import sys
import import_data
import bcolz
import pickle
import re
import evaluation
import random
import operator
from spellchecker import SpellChecker
from sklearn.model_selection import LeaveOneOut
import ast


torch.manual_seed(1)

# spell = SpellChecker()
# spell.word_frequency.add('scalene')
#
#
#
# glove_path = "./glove.6B"
#
# # Data
master_df = pd.read_csv("vectorized_data.csv", converters={0:ast.literal_eval, 1:ast.literal_eval}, )
#
# list_problems_individual_traditional = engage_ny.groupby('problem_id').count().reset_index()  ## THere are 113 problems
#
#
# # Create word vectors using GLOVE
# words = []
# idx = 0
# word2idx = {}
# vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.50.dat', mode='w')
#
# # with open(f'{glove_path}/glove.6B.50d.txt', 'rb') as f:
# #     for l in f:
# #         line = l.decode().split()
# #         word = line[0]
# #         words.append(word)
# #         word2idx[word] = idx
# #         idx += 1
# #         vect = np.array(line[1:]).astype(np.float)
# #         vectors.append(vect)
#
# vectors = bcolz.carray(vectors[1:].reshape((-1, 50)), rootdir=f'{glove_path}/6B.50.dat', mode='w')
# vectors.flush()
# pickle.dump(words, open(f'{glove_path}/6B.50_words.pkl', 'wb'))
# pickle.dump(word2idx, open(f'{glove_path}/6B.50_idx.pkl', 'wb'))
#
# vectors = bcolz.open(f'{glove_path}/6B.50.dat')[:]
# words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
# word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))
#
# # Dictionary to get vectors from... glove["the"] = vector(....)
# glove = {w: vectors[word2idx[w]] for w in words}
#
#
# def load_X_y(problem_id):
#     problem_object = engage_ny.loc[engage_ny['problem_id'] == problem_id]
#     answers = []
#     cleaned_answers = []
#     grades = []
#
#     # Try to clean up answers
#     for answer in problem_object.answer_text:
#         cleaned_answers.append(clean_answer(answer))
#
#     # Find max answer length for matrix
#     max_len = 0
#     for answer in cleaned_answers:
#         num_words = len(answer)
#         # print("Is", num_words, "greater than", max_len)
#         if num_words > max_len:
#             # print("yes, new max is", num_words)
#             max_len = num_words
#
#     # print("Max length:", max_len)
#     # print(cleaned_answers)
#
#     # Get each answer in order
#     # call the function to turn each answer to vectors
#     for answer in cleaned_answers:
#         # print(answer)
#         vector = convert_to_vector(answer, max_len)
#         # print(len(vector))
#         answers.append(vector)
#
#     # Get each grade in order
#     for grade in problem_object.correct:
#         # Create one hot encoding for grades
#         if grade == 0.0:
#             grades.append([1, 0, 0, 0, 0])
#         elif grade == 0.25:
#             grades.append([0, 1, 0, 0, 0])
#         elif grade == 0.50:
#             grades.append([0, 0, 1, 0, 0])
#         elif grade == 0.75:
#             grades.append([0, 0, 0, 1, 0])
#         elif grade == 1.0:
#             grades.append([0, 0, 0, 0, 1])
#
#
#     # return answers as list of vectors, and grades
#     # print(answers)
#     answers = torch.tensor(answers, dtype=torch.float)
#     # print(answers.shape)
#     grades = torch.tensor(grades, dtype=torch.float)
#
#
#     return answers, grades
#
#
#
# def clean_answer(sentence):
#     # print("Original sentence:\n", sentence)
#     sentence_words = list(re.sub("[^\w]", " ",  sentence).split())
#     # find those words that may be misspelled
#     misspelled = spell.unknown(sentence_words)
#     # print("Misspelled words:")
#     # print(misspelled)
#     new_sentence = sentence_words
#     for word in misspelled:
#         # print("Most likely", spell.correction(word))
#         # Get the one `most likely` answer
#         new_sentence = [w.replace(word,spell.correction(word)) for w in sentence_words]
#         # # Get a list of `likely` options
#         # print("likely", spell.candidates(word))
#
#     # print("Fixed sentence:\n", new_sentence)
#     return new_sentence
#
#
#
# def convert_to_vector(sentence, max_len):
#     # print("max", max_len)
#     answer = []
#     # Convert sentence to words
#     # words = re.sub("[^\w]", " ",  sentence).split()
#     # print("words length:", len(words))
#
#     for word in sentence:
#         try:
#             answer.append(glove[word])
#         except KeyError:
#             answer.append(np.random.normal(scale=0.6))
#
#     # Pad with zeroes if needed
#     if len(answer) < max_len:
#         answer.extend([random.random()] * (max_len - len(answer)))
#
#     # print(answer)
#     # Return vectorized answer
#     # print(len(answer))
#     return answer


# def get_vocabulary(problem_id):
#     word_list = []
#     problem_object = engage_ny.loc[engage_ny['problem_id'] == problem_id]
#
#     # for each response(row) in the problem object
#     for answer in problem_object.answer_text:
#         # Change answer to words
#         word_list.extend(re.sub("[^\w]", " ",  answer).split())
#
#     word_list = list(set(word_list))
#
#     matrix_len = len(word_list)
#     weights_matrix = np.zeros((matrix_len, 50))
#     words_found = 0
#
#     for i, word in enumerate(word_list):
#         try:
#             weights_matrix[i] = glove[word]
#             words_found += 1
#         except KeyError:
#             weights_matrix[i] = np.random.normal(scale=0.6, size=(50, ))
#
#     return weights_matrix, word_list

def load_problem_vectors(problem_id):

    problem_object = master_df.loc[master_df['problem_id'] == problem_id]


    answers = list(problem_object["answer"])
    grades = list(problem_object["grade"])

    # return answers as list of vectors, and grades
    # print(answers)
    answers = torch.tensor(answers, dtype=torch.float)
    # print(answers.shape)
    grades = torch.tensor(grades, dtype=torch.float)
    return answers, grades


def create_emb_layer(weights_matrix, non_trainable=False):
    # num_answers by max length (44x30)
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': torch.Tensor(weights_matrix)})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

class Neural_Network(nn.Module):
    def __init__(self, weights_matrix, hidden_size, num_layers):
        super().__init__()
        self.embedding, self.num_embeddings, self.embedding_dim = create_emb_layer(weights_matrix, True)
        self.hidden_size = hidden_size
        self.weights_matrix = weights_matrix
        self.output_size = 5
        self.num_layers = num_layers
        # weights
        self.W1 = torch.randn(self.embedding_dim, self.hidden_size) # 44 X 3 tensor
        self.W2 = torch.randn(self.hidden_size, self.output_size) # 3 X 1 tensor

    def forward(self, X):
        self.z = torch.matmul(X, self.W1)
        self.z2 = torch.tanh(self.z) # tanh activation function
        self.z3 = torch.matmul(self.z2, self.W2)
        m = nn.Softmax(dim=1)
        o = m(self.z3) # softmax activation function for output
        return o


    def backward(self, X, y, o):
        # self.o_error = torch.t(y) - o # error in output
        m = nn.Softmax(dim=1)
        self.o_error = y - o # error in output
        self.o_delta = self.o_error * m(o) # derivative of sig to error
        self.z2_error = torch.matmul(self.o_delta, torch.t(self.W2))
        self.z2_delta = self.z2_error * m(self.z2)
        self.W1 += torch.matmul(torch.t(X), self.z2_delta)
        self.W2 += torch.matmul(torch.t(self.z2), self.o_delta)

    def train(self, X, y):
        # forward + backward pass for training
        o = self.forward(X)
        self.backward(X, y, o)

    def predict(self):
        # print ("Predicted data based on trained weights: ")
        # print ("Input (scaled): \n" + str(self.weights_matrix))
        predictions = self.forward(self.weights_matrix)
        return predictions

    def multi_class_cross_entropy_loss(self, predictions, labels):
        """
        Calculate multi-class cross entropy loss for every pixel in an image, for every image in a batch.

        In the implementation,
        - the first sum is over all classes,
        - the second sum is over all rows of the image,
        - the third sum is over all columns of the image
        - the last mean is over the batch of images.

        :param predictions: Output prediction of the neural network.
        :param labels: Correct labels.
        :return: Computed multi-class cross entropy loss.
        """
        loss = -torch.mean(torch.sum(labels * torch.log(predictions), dim=1))
        return loss


def calculateGrade(question, response):
    answers, labels = load_X_y(question)
    cleaned_answer = clean_answer(response)
    answer = convert_to_vector(cleaned_answer, answers.shape[1])
    answer = torch.tensor([answer])
    try:
        NN = load_saved_model(question, answers)
    except FileNotFoundError:
        print("No training file")
        return 0
    # print("answer", answer.shape[1])
    # print("NN.embedding_dim before", NN.embedding_dim)
    # NN.embedding_dim = answer.shape[1]
    # print("NN.embedding_dim after", NN.embedding_dim)
    output = NN(answer)
    grade = group_by_bin(output)
    print("Grade:", grade)
    return grade



# Load trained model
def load_saved_model(problem_id, answers):
    return torch.load("Torch/trained_nn_" + str(problem_id) + ".pt")

# Save model
def save_weights(model, problem_id):
    torch.save(model, "Torch/trained_nn_" + str(problem_id) + ".pt")

# # Training
def run_training(count, problem_id, answers, labels):
    # 30, 44
    NN = Neural_Network(answers, 3, 3)
    # trains the NN 100 times
    for i in range(5):
        print ("#" + str(count) + " Loss: " + str(torch.mean((labels - NN.forward(answers))**2).detach().item()))  # mean sum squared loss
        NN.train(answers, labels)
    save_weights(NN, problem_id)

    # predictions = NN.predict()
    # groups = group_by_bin(predictions)
    return NN


def run_test(NN, problem_id, X_test):
    # try:
    #     NN = load_saved_model(problem_id, X_test)
    # except FileNotFoundError:
    #     print("No training file")
    #     return 0

    output = NN(X_test)
    grades = group_by_bin(output)
    return grades


def group_by_bin(output):
        # print("Predicted Grades")
        output = output.numpy()
        output = output[0]
        predicted_grades = []

        index, value = max(enumerate(output), key=operator.itemgetter(1))
        if index == 0:
            predicted_grades.append([1.0, 0.0, 0.0, 0.0, 0.0])
        if index == 1:
            predicted_grades.append([0.0, 1.0, 0.0, 0.0, 0.0])
        if index == 2:
            predicted_grades.append([0.0, 0.0, 1.0, 0.0, 0.0])
        if index == 3:
            predicted_grades.append([0.0, 0.0, 0.0, 1.0, 0.0])
        if index == 4:
            predicted_grades.append([0.0, 0.0, 0.0, 0.0, 1.0])

        if len(predicted_grades) == 1:
            return predicted_grades[0]
        else:
            return predicted_grades



        # for pred in output:
        #     if pred==0:
        #         # predicted_grades.append(0)
        #         predicted_grades.append([1, 0, 0, 0, 0])
        #     if pred >0 and pred<=0.25:
        #         # predicted_grades.append(0.25)
        #         predicted_grades.append([0, 1, 0, 0, 0])
        #     if pred>0.25 and pred<=0.50:
        #         # predicted_grades.append(0.50)
        #         predicted_grades.append([0, 0, 1, 0, 0])
        #     if pred>0.50 and pred<=0.75:
        #         # predicted_grades.append(0.75)
        #         predicted_grades.append([0, 0, 0, 1, 0])
        #     if pred>0.75 and pred<=1.0:
        #         predicted_grades.append([0, 0, 0, 0, 1])
        #         # predicted_grades.append(1.0)
        #
        # if len(predicted_grades) == 1:
        #     return predicted_grades[0]
        # else:
        #     return predicted_grades

def acc_vs_predicted(actual_grades, predicted_grades):
    actual_grades = actual_grades.tolist()
    actual_grades = actual_grades[0]
    # dict = {"Actual": actual_grades, "Predicted": predicted_grades}
    # df = pd.DataFrame(dict, columns=['Actual', 'Predicted'])
    # print(df)
    correct = 0

    # if len(actual_grades) > 1:
    #     actual_grades = actual_grades[0]
    # else:
    # for i in range(len(actual_grades)):
    # print("Actual:", actual_grades)
    # print("Predicted:", predicted_grades)
    # if actual_grades == predicted_grades:
    #     correct +=1
    # print("Percentage correct:", correct/len(actual_grades)*100, "%")
    auc = evaluation.auc(actual_grades, predicted_grades)
    # print("AUC score:", auc)
    rmse = evaluation.rmse(actual_grades, predicted_grades)
    # print("RMSE score:", rmse)
    kappa = evaluation.cohen_kappa(actual_grades, predicted_grades)
    # print("Cohen Kappa score:", kappa)
    multi_kappa = evaluation.cohen_kappa_multiclass(actual_grades, predicted_grades)
    # print("Cohen Kappa multiclass score:", kappa)

    return auc, rmse, kappa, multi_kappa

def train_test_per_problem(problem_id):
    answers, labels = load_problem_vectors(problem_id)
    # print(problem_id)

    # Run training with each set
    loo = LeaveOneOut()
    loo.get_n_splits(answers)

    auc_each_loop = []
    rmse_each_loop = []
    kappa_each_loop = []
    multi_kappa_each_loop = []

    average_auc = 0
    average_rmse = 0
    average_kappa = 0
    average_multi_kappa = 0
    count = 0

    NN = Neural_Network(answers, 3, 3)
    NN.cuda()
    # optimizer = optim.Adam(NN.parameters(), lr=0.0001)

    for train_index, test_index in loo.split(answers):
        # print("TRAIN:", train_index, "TEST:", test_index)
        count+=1
        X_train, X_test = answers[train_index], answers[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # NN = run_training(count, problem_id, X_train, y_train)
        # Train
        # print("y_train\n", y_train)

        print ("#" + str(count) + " Loss: " + str(NN.multi_class_cross_entropy_loss(NN.forward(X_train), y_train)))
        NN.train(X_train, y_train)

        # predictions = run_test(NN, problem_id, X_test)
        # Test
        output = NN(X_test)
        # print(output)
        output = output[0]

        # predictions = group_by_bin(output)

        # print(output)
        # optimizer.step()

        # Show how well we did with test set
        # y_test = group_by_bin(y_test.tolist()[0])

        auc, rmse, kappa, multi_kappa = acc_vs_predicted(y_test, output)
        auc_each_loop.append(auc)
        rmse_each_loop.append(rmse)
        kappa_each_loop.append(kappa)
        multi_kappa_each_loop.append(multi_kappa)

    # print("number of auc values:", len(auc_each_loop))
    # print("number of rmse values:", len(rmse_each_loop))

    save_weights(NN, problem_id)
    average_auc = sum(auc_each_loop) / len(auc_each_loop)
    average_rmse = sum(rmse_each_loop) / len(rmse_each_loop)
    average_kappa = sum(kappa_each_loop) / len(kappa_each_loop)
    average_multi_kappa = sum(multi_kappa_each_loop) / len(multi_kappa_each_loop)

    # print("number of AUC values:", len(overall_auc))
    # print("number of rmse values:", len(overall_rmse))
    print("Average AUC:", average_auc)
    print("Average RMSE:", average_rmse)
    print("Average Kappa:", average_kappa)
    print("Average Multi Kappa:", average_multi_kappa)

    return average_auc, average_rmse, average_kappa, average_multi_kappa

def train_test_all_problems():
    overall_auc = []
    final_auc = 0
    overall_rmse = []
    final_rmse = 0
    overall_kappa = []
    final_kappa = 0
    overall_multi_kappa = []
    final_multi_kappa = 0
    count = 0

    all_problems = master_df.groupby('problem_id').count().reset_index()

    # test_problem_ids = [36600, 36601]
    # list_problems_individual_traditional["problem_id"]

    for pid in all_problems["problem_id"]:
    # for pid in test_problem_ids:
        count+=1
        print("Problem: " + str(count) +"/" + str(len(all_problems["problem_id"])) + ": " + str(pid))
        average_auc, rmse_each_loop, kappa_each_loop, multi_kappa_each_loop = train_test_per_problem(pid)
        overall_auc.append(average_auc)
        overall_rmse.append(rmse_each_loop)
        overall_kappa.append(kappa_each_loop)
        overall_multi_kappa.append(multi_kappa_each_loop)

    final_auc = sum(overall_auc) / len(overall_auc)
    final_rmse = sum(overall_rmse) / len(overall_rmse)
    final_kappa = sum(overall_kappa) / len(overall_kappa)
    final_multi_kappa = sum(overall_multi_kappa) / len(overall_multi_kappa)

    print("Final AUC:", final_auc)
    print("Final RMSE:", final_rmse)
    print("Final Kappa:", final_kappa)
    print("Final Multi Kappa:", final_multi_kappa)

    # 7/25 (changed to softmax)
    # Final AUC: 0.7665351055857251
    # Final RMSE: 0.23624986082472338
    # Final Kappa: 0.5330702111714496
    # Final Multi Kappa: 1.0

    # 7/26 (after fixing the probabilities/binning)
    # Final AUC: 0.7767902188616805
    # Final RMSE: 0.2505419710498344
    # Final Kappa: 0.4971732283353235
    # Final Multi Kappa: 1.0

# Problem ids
# # 1487480 - 43 answers
if __name__ == '__main__':
    # problem_id = int(sys.argv[1])
    # weights_matrix, words_list = get_vocabulary(problem_id)
    train_test_all_problems()
    # train_test_per_problem(1487480)


    # calculateGrade(problem_id, "No")
    # clean_answer("It has 2 lines that are parrellel to eachother and perpendicular to the planes.")

