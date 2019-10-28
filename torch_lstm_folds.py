import sklearn
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import sys

from torch.nn.utils.rnn import pack_padded_sequence

import import_data
import bcolz
import pickle
import re
import evaluation
import random
import operator
from spellchecker import SpellChecker
from sklearn.model_selection import LeaveOneOut, RepeatedKFold
from sklearn.model_selection import KFold
import datetime

torch.manual_seed(1)


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

# Hyper-parameters
# sequence_length = 20
hidden_size = 20
num_layers = 1
num_classes = 5
batch_size = 1
num_epochs = 2
learning_rate = 0.003

embedding_length = 100



def load_problem_vectors(problem_id, master_df):
    # # Data
    # master_df = pd.read_csv("vectorized_" + dataset_name + "_no_spellcheck.csv", converters={1: ast.literal_eval, 2: ast.literal_eval})

    problem_object = master_df.loc[master_df['problem_id'] == problem_id]

    answers = []

    for ans in list(problem_object["answer"]):

        if ans == []:
            answers.append(torch.LongTensor([0]).to(device))
        else:
            if len(ans) > 1000:
                print("ans is", len(ans))
                answers.append(torch.LongTensor(ans[:1000]).to(device))
            else:
                answers.append(torch.LongTensor(ans).to(device))

    # answers = list(problem_object["answer"])
    grades = list(problem_object["grade"])
    folds = list(problem_object["folds"])
    problem_log_ids = list(problem_object["problem_log_id"])
    grader_teacher_ids = list(problem_object["grader_teacher_id"])

    # return answers as list of vectors, and grades
    # answers = torch.tensor(answers, dtype=torch.long)#, requires_grad=True)
    # grades = torch.tensor(grades, dtype=torch.long)
    return answers, grades, problem_log_ids, grader_teacher_ids, folds


class LSTMClassifier(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, num_layers, vocab_size, embedding_length, weights):
        super(LSTMClassifier, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of the batch
        output_size : 5 = [0,0,0,0,1]
        hidden_size : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 

        """

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embedding_length = embedding_length

        self.word_embeddings = nn.Embedding(vocab_size, embedding_length).to(device)  # Initializing the look-up table.
        self.word_embeddings.weight = nn.Parameter(torch.tensor(weights), requires_grad=False)  # Assigning the look-up table to the pre-trained GloVe word embedding.

        self.lstm = nn.LSTM(embedding_length, hidden_size, batch_first=True, bidirectional=True).to(device)
        self.label = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax()


    def forward(self, input_sentence, batch_size, ans_lengths):
        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for positive & negative class which receives its input as the final_hidden_state of the LSTM
        final_output.shape = (batch_size, output_size)
        """
        input_embed = self.word_embeddings(input_sentence)  # embedded input of shape = (batch_size, num_sequences,  embedding_length)
        # pack the embedded input so that lstm doesnt use padded 0s
        x_packed = pack_padded_sequence(input_embed, ans_lengths, batch_first=True, enforce_sorted=False)

        if batch_size is None:
            # Set initial states
            h_0 = torch.zeros(self.num_layers * 2, input_sentence.size(0), self.hidden_size).to(device)  # 2 for bidirection
            c_0 = torch.zeros(self.num_layers * 2, input_sentence.size(0), self.hidden_size).to(device)
        else:
            h_0 = torch.zeros(self.num_layers * 2, input_sentence.size(0), self.hidden_size).to(device)
            c_0 = torch.zeros(self.num_layers * 2, input_sentence.size(0), self.hidden_size).to(device)

        output, (final_hidden_state, final_cell_state) = self.lstm(x_packed, (h_0, c_0))
        label_output = self.label(final_hidden_state[-1])  # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)
        final_output = self.softmax(label_output)
        return final_output


def calculate_grade(question, response):
    cleaned_answer = import_data.clean_answer(response)
    try:
        with open('./glove.6B/dataset_w2idx.pkl', 'rb') as f:
            dataset_word2idx = pickle.load(f)
    except FileNotFoundError:
        print("No idx file")

    answer = import_data.convert_to_idx_vector(dataset_word2idx, cleaned_answer)
    answer = torch.tensor([answer]).to(device)
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
    return torch.load("LSTM/trained_lstm_" + str(problem_id) + ".pt")

# Save model
def save_weights(model, problem_id):
    torch.save(model, "LSTM/trained_lstm_" + str(problem_id) + ".pt")



def group_by_bin(output):
        # print("Predicted Grades")
        # output = output.numpy()
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
    # print(predicted_grades)
    rounded = group_by_bin(predicted_grades)
    # predicted_grades = predicted_grades.tolist()

    # actual_grades = actual_grades[0]

    # print("actual", actual_grades)
    # print("predicted", rounded)
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
    # print(actual_grades)
    # print(predicted_grades)
    auc = evaluation.auc(actual_grades, predicted_grades)
    # print("AUC score:", auc)

    # Feed RMSE an (n by 1)
    rmse = evaluation.rmse(rmse_actual, rmse_predicted)
    kappa = evaluation.cohen_kappa(actual_grades, predicted_grades)
    multi_kappa = evaluation.cohen_kappa_multiclass(actual_grades, predicted_grades)

    return auc, rmse, kappa, multi_kappa

def get_mean_std_each_class(output):
    output = output.detach().numpy()
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

    # z_scores = z_scores.squeeze(axis=1)

    return torch.tensor(z_scores, dtype=torch.float)



def train_test_per_problem(problem_id, prob_number, all_prob_len, dataset_name, master_df, vocab_list, weights_matrix):
    answers, labels, problem_log_ids, grader_teacher_ids, folds = load_problem_vectors(problem_id, master_df)

    all_fold_numbers = []

    for fold in folds:
        if fold not in all_fold_numbers:
            all_fold_numbers.append(fold)

    print("All fold numbers:", all_fold_numbers)
    final_predictions = []
    test_order = []

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

    vocab_length = len(vocab_list)
    # answers = torch.tensor(answers, dtype=torch.long)  # , requires_grad=True)
    seq_lengths = torch.LongTensor([len(ans) for ans in answers]).to(device)
    answers = pad_sequence(answers, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)

    answers = answers.to(device)
    print(answers.shape)
    labels = labels.to(device)

    test_fold = True

    run = 0
    for f in range(1, 11):
        test_fold = True
        # if there are only 1 folds in the set
        if len(all_fold_numbers) == 1:
            test_fold = False
            for i in range(len(answers)):
                all_test_outputs.append([0,0,0,0,1])
            break

        if f not in all_fold_numbers:
            continue

        validation_fold = False
        val_index = []
        epochs = 100

        # if there are only 2 folds in the set
        if len(all_fold_numbers) == 2:
            test_index = [i for i, x in enumerate(folds) if x == f]
            train_index = [i for i, x in enumerate(folds) if x != f]
            epochs = 60

        # if there are at least 3 fold numbers, we can use one for validation
        if len(all_fold_numbers) > 2:
            validation_fold = True
            test_index = [i for i, x in enumerate(folds) if x == f]
            for val in folds:
                if val != f:
                    validation_fold_num = val
                    break
            val_index = [i for i, x in enumerate(folds) if x == validation_fold_num]
            train_index = [i for i, x in enumerate(folds) if x != f and x != validation_fold_num]


        run+=1
        print("Problem:", str(prob_number), "/", all_prob_len)
        print("Training fold:", f, "/", 10)

        print("TRAIN:", train_index, "TEST:", test_index, "VAL:", val_index)

        X_train, X_test = answers[train_index], answers[test_index]
        X_train_seqs = seq_lengths[train_index]
        X_test_seqs = seq_lengths[test_index]
        y_train, y_test = labels[train_index], labels[test_index]


        if validation_fold:
            X_validation, y_validation = answers[val_index], labels[val_index]
            X_val_seqs = seq_lengths[val_index]

        # Reshape everything to have 3 dims
        answers = answers.reshape(-1, answers.shape[1]).to(device)
        X_train = X_train.reshape(-1, answers.shape[1]).to(device)
        X_test = X_test.reshape(-1, answers.shape[1]).to(device)


        # Free up gpu memory
        torch.cuda.empty_cache()

        model = LSTMClassifier(batch_size, num_classes, hidden_size, num_layers, vocab_length, embedding_length, weights_matrix)
        model = model.to(device)
            # (input_size, hidden_size, num_layers, num_classes).to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        count = 0

        # Train
        for i in range(epochs):

            output = model.forward(X_train, 1, X_train_seqs)

            y_train = y_train.long()
            loss = criterion(output, torch.max(y_train, 1)[1])

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            count += 1
            print ("#" + str(count) + " Loss: " + str(loss))

            # val is low but slightly higher than training loss is ideal
            if validation_fold:
                output_validation = model.forward(X_validation, 1, X_val_seqs)
                loss_val = criterion(output_validation, torch.max(y_validation, 1)[1])
                loss_val.backward()
                optimizer.step()
                # print("Validation loss is:", loss_val.item(), "Training loss is:", loss.item())
                if round(loss_val.item(), 4) >= round(loss.item(), 4):
                    break


        # Get the mean and std from the training set

        # train_predictions = model(answers)
        # # _, train_predictions = torch.max(test_output.data, 1)
        # train_predictions = train_predictions.cpu()
        # train_predictions = train_predictions.squeeze(0)

        # Save weights
        save_weights(model, problem_id)
        # col_means, col_stds = get_mean_std_each_class(train_predictions)
        # all_col_means.append(col_means)
        # all_col_stds.append(col_stds)

        # Test
        test_output = model(X_test, 1, X_test_seqs)
        all_test_outputs.extend(test_output.tolist())

        test_order.extend(test_index)

        # Reshape answers back for split
        # answers = answers.reshape(answers_len, sequence_length).to(device)
        answers = answers.squeeze(0)
        labels = labels.squeeze(0)


    if test_fold:
        sorted_tests = [x for _, x in sorted(zip(test_order, all_test_outputs))]
        all_test_outputs = sorted_tests

    auc, rmse, kappa, multi_kappa = acc_vs_predicted(labels, all_test_outputs)



    # z_score_output = z_score_output.numpy()

    # print(test_output)
    # Save test output to csv
    # for i in range(z_score_output.shape[0]):
    #     pred = np.insert(z_score_output[i], 0, problem_log_ids[i])
    #     pred = pred.tolist()
    #     final_predictions.append(pred)

    for i in range(len(all_test_outputs)):
        pred = np.insert(all_test_outputs[i], 0, problem_log_ids[i])
        pred2 = np.insert(pred, 1, grader_teacher_ids[i])
        pred2 = pred2.tolist()
        final_predictions.append(pred2)

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

def train_test_all_problems(master_df, all_problems, dataset_name, vocab_list, weights_matrix):
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

    #test_problem_ids = [1070912] #has one response
    # list_problems_individual_traditional["problem_id"]

    for pid in all_problems["problem_id"]:
    #for pid in test_problem_ids:
        count+=1
        all_prob_len = str(len(all_problems["problem_id"]))
        print("/////////////////////////// Problem: " + str(count) +"/" + all_prob_len + ": " + str(pid))
        average_auc, rmse_each_loop, kappa_each_loop, multi_kappa_each_loop, final_predictions_each_loop= train_test_per_problem(pid, count, all_prob_len, dataset_name, master_df, vocab_list, weights_matrix)

        # Append to final pred dataframe
        if final_predictions_each_loop.empty == False:
            all_final_predictions = all_final_predictions.append(final_predictions_each_loop)
        print(all_final_predictions)


        if (average_auc != 0):
            overall_auc.append(average_auc)
        if rmse_each_loop != -1 or kappa_each_loop !=-1 or multi_kappa_each_loop !=-1:
            overall_rmse.append(rmse_each_loop)
            overall_kappa.append(kappa_each_loop)
            overall_multi_kappa.append(multi_kappa_each_loop)

    final_auc = sum(overall_auc) / len(overall_auc)
    final_rmse = sum(overall_rmse) / len(overall_rmse)
    final_kappa = sum(overall_kappa) / len(overall_kappa)
    final_multi_kappa = sum(overall_multi_kappa) / len(overall_multi_kappa)

    all_final_predictions.to_csv("final_predictions_lstm_glove3.csv")

    print("Final AUC:", final_auc)
    print("Final AUC length:", len(overall_auc))
    print("Final RMSE:", final_rmse)
    print("Final Kappa:", final_kappa)
    print("Final Multi Kappa:", final_multi_kappa)

    now = datetime.datetime.now()
    f = open("Results/"+ dataset_name +"/eval-lstm-" + str(now.strftime("%Y-%m-%d")) + str(now.microsecond) + ".txt", "w+")
    f.write("Final AUC: " + str(final_auc))
    f.write("\nFinal RMSE: " + str(final_rmse))
    f.write("\nFinal Kappa: " + str(final_kappa))
    f.write("\nFinal  Multi Kappa:" + str(final_multi_kappa))
    f.close()

    print(all_final_predictions)




# Problem ids
# # 1487480 - 43 answers
if __name__ == '__main__':

    calculate_grade(41099, "I like cats")

