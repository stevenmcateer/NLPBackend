import sklearn
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
import datetime

torch.manual_seed(1)

#

def load_problem_vectors(problem_id, master_df):
    # # Data
    # master_df = pd.read_csv("vectorized_" + dataset_name + "_no_spellcheck.csv", converters={1: ast.literal_eval, 2: ast.literal_eval})

    problem_object = master_df.loc[master_df['problem_id'] == problem_id]

    answers = []
    # Trim answers to max length of 20 words
    for ans in list(problem_object["answer"]):
        answers.append(ans[:20])


    grades = list(problem_object["grade"])
    problem_log_ids = list(problem_object["problem_log_id"])

    # return answers as list of vectors, and grades
    answers = torch.tensor(answers, dtype=torch.float)
    grades = torch.tensor(grades, dtype=torch.long)
    return answers, grades, problem_log_ids


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
        # self.dropout = nn.Dropout(p=0.5)
        self.num_layers = num_layers
        # weights
        self.W1 = torch.randn(self.embedding_dim, self.hidden_size)
        self.W2 = torch.randn(self.hidden_size, self.output_size)

    def forward(self, X):
        self.z = torch.matmul(X, self.W1)
        self.z2 = torch.relu(self.z) # relu activation function
        # self.z2 = self.dropout(self.z2)
        self.z3 = torch.matmul(self.z2, self.W2)
        m = nn.Softmax(dim=1)
        o = m(self.z3) # softmax activation function for output
        return o


    def backward(self, X, y, o):
        # self.o_error = torch.t(y) - o # error in output
        m = nn.Softmax(dim=1)
        y = y.float()
        self.o_error = y - o # error in output
        self.o_delta = self.o_error * m(o)
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


def calculate_grade(question, response):
    max_length = 20
    cleaned_answer = import_data.clean_answer(response)
    answer = import_data.convert_to_vector(cleaned_answer, max_length)
    answer = torch.tensor([answer])
    try:
        NN = load_saved_model(question, answer)
    except FileNotFoundError:
        print("No training file")
        return 0

    output = NN(answer)
    print("Grade:\n", output)
    return output



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
        predicted_grades = []
        for pred in output:
            index, value = max(enumerate(pred), key=operator.itemgetter(1))
            if index == 0:
                predicted_grades.append([1, 0, 0, 0, 0])
            if index == 1:
                predicted_grades.append([0, 1, 0, 0, 0])
            if index == 2:
                predicted_grades.append([0, 0, 1, 0, 0])
            if index == 3:
                predicted_grades.append([0, 0, 0, 1, 0])
            if index == 4:
                predicted_grades.append([0, 0, 0, 0, 1])


        return predicted_grades

def acc_vs_predicted(actual_grades, predicted_grades):
    actual_grades = actual_grades.tolist()
    rounded = group_by_bin(predicted_grades)
    predicted_grades = predicted_grades.tolist()

    # actual_grades = actual_grades[0]
    dict = {"Actual": actual_grades, "Predicted": rounded}
    df = pd.DataFrame(dict, columns=['Actual', 'Predicted'])
    print(df)
    correct = 0


    rmse_actual = []
    rmse_predicted = []
    # if len(actual_grades) > 1:
    #     actual_grades = actual_grades[0]
    # else:
    for i in range(len(actual_grades)):
        if actual_grades[i] == rounded[i]:
            correct +=1

        # populate rmse lists
        rmse_actual.append(np.argmax(actual_grades[i]))
        rmse_predicted.append(np.argmax(predicted_grades[i]))


    print("Percentage correct:", correct/len(actual_grades)*100, "%")
    auc = evaluation.auc(actual_grades, predicted_grades)
    # print("AUC score:", auc)

    # Feed RMSE an (n by 1)
    rmse = evaluation.rmse(rmse_actual, rmse_predicted)
    kappa = evaluation.cohen_kappa(actual_grades, predicted_grades)
    multi_kappa = evaluation.cohen_kappa_multiclass(actual_grades, predicted_grades)

    return auc, rmse, kappa, multi_kappa

def get_mean_std_each_class(output):
    output = output.numpy()
    col_means = []
    col_stds = []

    # for each class in the training predictions, calc mean and std
    for i in range(0, 5):
        # train_predictions[:, i] = sum(train_predictions[:, i])/len(train_predictions[:, i])
        col_means.append(np.mean(output[:, i]))
        col_stds.append(np.std(output[:, i]))

    return col_means, col_stds

def convert_to_z_score(output, col_means, col_stds):
    output = np.asarray(output)
    z_scores = np.zeros(output.shape)
    # norm_scores = np.zeros(output.shape)

    for c in range(z_scores.shape[1]):
        for r in range(z_scores.shape[0]):
            # print(output[r,c])
            if col_stds[c] == 0.0:
                z_scores[r, c] = 0.0
            else:
                z_scores[r, c] = (output[r, c] - col_means[c]) / col_stds[c]


    return torch.tensor(z_scores, dtype=torch.float)



def train_test_per_problem(problem_id, dataset_name, master_df):
    answers, labels, problem_log_ids = load_problem_vectors(problem_id, master_df)
    # print(problem_id)

    # Run training with each set
    loo = LeaveOneOut()
    loo.get_n_splits(answers)

    final_predictions = []

    auc_each_loop = []
    rmse_each_loop = []
    kappa_each_loop = []
    multi_kappa_each_loop = []

    all_test_outputs = []
    all_col_means = []
    all_col_stds = []

    average_auc = 0
    average_rmse = 0
    average_kappa = 0
    average_multi_kappa = 0

    # NN = Neural_Network(answers, 5, 3)
    # NN.cuda()

    # print(NN.parameters())
    # Free up gpu memory
    torch.cuda.empty_cache()
    split = 0
    for train_index, test_index in loo.split(answers):
        # print("TRAIN:", train_index, "TEST:", test_index)
        split+=1
        print("Training current split:", split, "/", len(answers))
        # add check for the leave one out 1 response case, just use the one
        if len(answers) != 1:
            X_train, X_test = answers[train_index], answers[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
        else:
            X_train, X_test = answers[test_index], answers[test_index]
            y_train, y_test = labels[test_index], labels[test_index]



        NN = Neural_Network(X_train, 5, 3)
        NN.cuda()
        # NN.train(True)
        optimizer = optim.Adam(NN.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        count = 0

        # Train
        for i in range(1):
            output = NN.forward(X_train)

            output.requires_grad = True

            y_train = y_train.long()
            loss = criterion(output, torch.max(y_train, 1)[1])
            # loss.requires_grad = True
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            count += 1
            print ("#" + str(count) + " Loss: " + str(loss))

            NN.train(X_train, y_train)

        # Get the mean and std from the training set
        train_predictions = NN.predict()

        # Save weights
        save_weights(NN, problem_id)
        col_means, col_stds = get_mean_std_each_class(train_predictions)
        all_col_means.append(col_means)
        all_col_stds.append(col_stds)

        # Test
        test_output = NN(X_test)
        all_test_outputs.append(test_output.tolist()[0])


    all_col_means = np.asarray(all_col_means)
    all_col_stds = np.asarray(all_col_stds)

    final_means = []
    final_stds = []
    # for each class in the training predictions, calc mean and std
    for i in range(0, 5):
        final_means.append(np.mean(all_col_means[:, i]))
        final_stds.append(np.std(all_col_stds[:, i]))

    test_output = convert_to_z_score(all_test_outputs, final_means, final_stds)

    auc, rmse, kappa, multi_kappa = acc_vs_predicted(labels, test_output)

    test_output = test_output.numpy()
    # print(test_output)
    # Save test output to csv
    for i in range(test_output.shape[0]):
        pred = np.insert(test_output[i], 0, problem_log_ids[i])
        pred = pred.tolist()
        final_predictions.append(pred)

    final_predictions = np.asarray(final_predictions)
    final_pred_df = pd.DataFrame(final_predictions)

    if (np.isnan(auc)):
        print("AUC is nan, not appending anything")
        # auc_each_loop.append(0.5)
    else:
        auc_each_loop.append(auc)
    rmse_each_loop.append(rmse)
    kappa_each_loop.append(kappa)
    multi_kappa_each_loop.append(multi_kappa)

    print("number of auc values:", len(auc_each_loop))


    if len(auc_each_loop) > 0:
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

    return average_auc, average_rmse, average_kappa, average_multi_kappa, final_pred_df

def train_test_all_problems(master_df, all_problems, dataset_name):
    overall_auc = []
    final_auc = 0
    overall_rmse = []
    final_rmse = 0
    overall_kappa = []
    final_kappa = 0
    overall_multi_kappa = []
    final_multi_kappa = 0
    count = 0

    all_final_predictions = pd.DataFrame()

    # test_problem_ids = [36600, 36601]
    # list_problems_individual_traditional["problem_id"]

    for pid in all_problems["problem_id"]:
    # for pid in test_problem_ids:
        count+=1
        print("Problem: " + str(count) +"/" + str(len(all_problems["problem_id"])) + ": " + str(pid))
        average_auc, rmse_each_loop, kappa_each_loop, multi_kappa_each_loop, final_predictions_each_loop= train_test_per_problem(pid, dataset_name, master_df)

        # Append to final pred dataframe
        all_final_predictions = all_final_predictions.append(final_predictions_each_loop)


        if (average_auc != 0):
            overall_auc.append(average_auc)
        overall_rmse.append(rmse_each_loop)
        overall_kappa.append(kappa_each_loop)
        overall_multi_kappa.append(multi_kappa_each_loop)

    final_auc = sum(overall_auc) / len(overall_auc)
    final_rmse = sum(overall_rmse) / len(overall_rmse)
    final_kappa = sum(overall_kappa) / len(overall_kappa)
    final_multi_kappa = sum(overall_multi_kappa) / len(overall_multi_kappa)

    print("Final AUC:", final_auc)
    print("Final AUC length:", len(overall_auc))
    print("Final RMSE:", final_rmse)
    print("Final Kappa:", final_kappa)
    print("Final Multi Kappa:", final_multi_kappa)

    now = datetime.datetime.now()
    f = open("Results/"+ dataset_name +"/eval-torch-" + str(now.strftime("%Y-%m-%d")) + str(now.microsecond) + ".txt", "w+")
    f.write("Final AUC: " + str(final_auc))
    f.write("\nFinal RMSE: " + str(final_rmse))
    f.write("\nFinal Kappa: " + str(final_kappa))
    f.write("\nFinal  Multi Kappa:" + str(final_multi_kappa))
    f.close()


    all_final_predictions.to_csv("final_predictions_torch.csv")


# Problem ids
# # 1487480 - 43 answers
if __name__ == '__main__':
    # problem_id = int(sys.argv[1])
    # weights_matrix, words_list = get_vocabulary(problem_id)
    # train_test_all_problems()
    # train_test_per_problem(1487480)
    # sample = torch.tensor([[-2.2767, 4.1321, 2.7677, -5.0111, 2.9111],
    #                       [6.2224, 1.4400, -0.4112, 2.4321, 0.3112]])
    # print(sample)
    # convert_to_z_score(sample)

    calculate_grade(41099, "There must be 2 because I said so.")
    # clean_answer("It has 2 lines that are parrellel to eachother and perpendicular to the planes.")

