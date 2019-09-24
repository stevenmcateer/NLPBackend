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


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# Hyper-parameters
sequence_length = 20
input_size = 20
hidden_size = 20
num_layers = 1
num_classes = 5
batch_size = 1
num_epochs = 2
learning_rate = 0.003

embedding_length = 50



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
    # answers = torch.tensor(answers, dtype=torch.long)#, requires_grad=True)
    # grades = torch.tensor(grades, dtype=torch.long)
    return answers, grades, problem_log_ids

# def create_emb_layer(weights_matrix, non_trainable=False):
#
#     num_embeddings, embedding_dim = weights_matrix.shape
#     emb_layer = nn.Embedding(num_embeddings, embedding_dim)
#     emb_layer.load_state_dict({'weight': torch.Tensor(weights_matrix)})
#     if non_trainable:
#         emb_layer.weight.requires_grad = False
#
#     return emb_layer, num_embeddings, embedding_dim

class LSTMClassifier(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, num_layers, vocab_size, embedding_length, weights):
        super(LSTMClassifier, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
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

        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)  # Initializing the look-up table.
        self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)  # Assigning the look-up table to the pre-trained GloVe word embedding.


        self.lstm = nn.LSTM(embedding_length, hidden_size, batch_first=True, bidirectional=True)
        self.label = nn.Linear(hidden_size, output_size)

    def forward(self, input_sentence, batch_size=None):

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

        ''' Here we will map all the indexes present in the input sequence to the corresponding word vector using our pre-trained word_embedddins.'''
        # print(input_sentence)

        input = self.word_embeddings(input_sentence)  # embedded input of shape = (batch_size, num_sequences,  embedding_length)
        # input = input.permute(1, 0, 2)  # input.size() = (num_sequences, batch_size, embedding_length)
        if batch_size is None:
            h_0 = torch.zeros(self.num_layers * 2, input_sentence.size(0), self.hidden_size).to(device)  # Initial hidden state of the LSTM
            c_0 = torch.zeros(self.num_layers * 2, input_sentence.size(0), self.hidden_size).to(device)  # Initial cell state of the LSTM
            # # Set initial states
            # h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # 2 for bidirection
            # c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        else:
            h_0 = torch.zeros(self.num_layers * 2, input_sentence.size(0), self.hidden_size).to(device)
            c_0 = torch.zeros(self.num_layers * 2, input_sentence.size(0), self.hidden_size).to(device)
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        final_output = self.label(final_hidden_state[-1])  # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)

        return final_output


def calculate_grade(question, response):
    max_length = 20
    cleaned_answer = import_data.clean_answer(response)
    try:
        with open('./glove.6B/dataset_w2idx.pkl', 'rb') as f:
            dataset_word2idx = pickle.load(f)
    except FileNotFoundError:
        print("No idx file")

    answer = import_data.convert_to_idx_vector(dataset_word2idx, cleaned_answer, max_length)
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



def train_test_per_problem(problem_id, dataset_name, master_df, vocab_list, weights_matrix):
    answers, labels, problem_log_ids = load_problem_vectors(problem_id, master_df)
    # print(answers)

    # Run training with each set
    loo = LeaveOneOut()
    # loo.get_n_splits(answers)

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


    validation_split = .2
    shuffle_dataset = True
    random_seed = 42


    # Creating data indices for training and validation splits:
    dataset_size = len(answers)
    print(dataset_size)
    indices = list(range(dataset_size))
    if dataset_size == 1:
        split = 0
    elif dataset_size <= 4 and dataset_size != 1:
        split = 1
    else:
        split = int(np.floor(validation_split * dataset_size))
        print(np.floor(validation_split * dataset_size))
    print(split)
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    print("Train indices,", train_indices, "val indices,", val_indices)
    vocab_length = len(vocab_list)
    answers = torch.tensor(answers, dtype=torch.long)  # , requires_grad=True)
    labels = torch.tensor(labels, dtype=torch.long)

    X_train_chunk = answers[train_indices]
    y_train_chunk = labels[train_indices]



    answers = answers.to(device)
    labels = labels.to(device)
    X_train_chunk = X_train_chunk.to(device)
    y_train_chunk = y_train_chunk.to(device)

    run = 0
    for train_index, test_index in loo.split(X_train_chunk):

        run+=1
        print("Training current split:", run, "/", len(X_train_chunk))
        # add check for the leave one out 1 response case, just use the one
        if len(X_train_chunk) != 1:
            print("TRAIN:", train_index, "TEST:", test_index, "VAL:", val_indices)

            X_train, X_test = X_train_chunk[train_index], X_train_chunk[test_index]
            y_train, y_test = y_train_chunk[train_index], y_train_chunk[test_index]
            X_validation, y_validation = answers[val_indices], labels[val_indices]
        else:
            print("TRAIN:", test_index, "TEST:", test_index, "VAL:", test_index)
            X_train, X_test = X_train_chunk[test_index], X_train_chunk[test_index]
            y_train, y_test = y_train_chunk[test_index], y_train_chunk[test_index]
            X_validation, y_validation = X_train_chunk[test_index], y_train_chunk[test_index]


        # Reshape everything to have 3 dims
        answers = answers.reshape(-1, sequence_length).to(device)
        X_train = X_train.reshape(-1, sequence_length).to(device)
        X_test = X_test.reshape(-1, sequence_length).to(device)
        X_validation = X_validation.reshape(-1, sequence_length).to(device)
        # print(X_validation)


        # Free up gpu memory
        torch.cuda.empty_cache()

        model = LSTMClassifier(batch_size, num_classes, hidden_size, num_layers, vocab_length, embedding_length, weights_matrix)
        model = model.to(device)
            # (input_size, hidden_size, num_layers, num_classes).to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        count = 0
        epochs = 100
        # Train
        for i in range(epochs):

            output = model.forward(X_train)

            y_train = y_train.long()
            y_validation = y_validation.long()

            # print("Comparing", output, "to ", torch.max(y_train, 1)[1])
            # loss = criterion(output, y_train)
            loss = criterion(output, torch.max(y_train, 1)[1])

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            count += 1
            print ("#" + str(count) + " Loss: " + str(loss))

            # val is low but slightly higher than training loss is ideal
            if epochs % 20 == 0:
                output_validation = model.forward(X_validation)
                loss_val = criterion(output_validation, torch.max(y_validation, 1)[1])
                loss_val.backward()
                optimizer.step()
                print("Validation loss is:", loss_val.item(), "Training loss is:", loss.item())
                if round(loss_val.item(), 4) >= round(loss.item(), 4):
                    break


        # Get the mean and std from the training set

        train_predictions = model(answers)
        # _, train_predictions = torch.max(test_output.data, 1)
        train_predictions = train_predictions.cpu()
        # train_predictions = train_predictions.squeeze(0)

        # Save weights
        save_weights(model, problem_id)
        col_means, col_stds = get_mean_std_each_class(train_predictions)
        all_col_means.append(col_means)
        all_col_stds.append(col_stds)

        # Test
        test_output = model(X_test)
        all_test_outputs.append(test_output.tolist()[0])

        # Reshape answers back for split
        # answers = answers.reshape(answers_len, sequence_length).to(device)
        answers = answers.squeeze(0)
        labels = labels.squeeze(0)

    all_col_means = np.asarray(all_col_means)
    all_col_stds = np.asarray(all_col_stds)

    final_means = []
    final_stds = []
    # for each class in the training predictions, calc mean and std
    for i in range(0, 5):
        final_means.append(np.mean(all_col_means[:, i]))
        final_stds.append(np.std(all_col_stds[:, i]))

    # z_score_output = convert_to_z_score(all_test_outputs, final_means, final_stds)

    # auc, rmse, kappa, multi_kappa = acc_vs_predicted(labels, z_score_output)
    print(y_train_chunk)
    auc, rmse, kappa, multi_kappa = acc_vs_predicted(y_train_chunk, all_test_outputs)



    # z_score_output = z_score_output.numpy()

    # print(test_output)
    # Save test output to csv
    # for i in range(z_score_output.shape[0]):
    #     pred = np.insert(z_score_output[i], 0, problem_log_ids[i])
    #     pred = pred.tolist()
    #     final_predictions.append(pred)

    for i in range(len(all_test_outputs)):
        pred = np.insert(all_test_outputs[i], 0, problem_log_ids[i])
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

    # test_problem_ids = [44010] has one response
    # list_problems_individual_traditional["problem_id"]

    for pid in all_problems["problem_id"]:
    # for pid in test_problem_ids:
        count+=1
        print("Problem: " + str(count) +"/" + str(len(all_problems["problem_id"])) + ": " + str(pid))
        average_auc, rmse_each_loop, kappa_each_loop, multi_kappa_each_loop, final_predictions_each_loop= train_test_per_problem(pid, dataset_name, master_df, vocab_list, weights_matrix)

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
    f = open("Results/"+ dataset_name +"/eval-lstm-" + str(now.strftime("%Y-%m-%d")) + str(now.microsecond) + ".txt", "w+")
    f.write("Final AUC: " + str(final_auc))
    f.write("\nFinal RMSE: " + str(final_rmse))
    f.write("\nFinal Kappa: " + str(final_kappa))
    f.write("\nFinal  Multi Kappa:" + str(final_multi_kappa))
    f.close()


    all_final_predictions.to_csv("final_predictions_lstm_new.csv")


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

    calculate_grade(41099, "I like cats")
    # clean_answer("It has 2 lines that are parrellel to eachother and perpendicular to the planes.")

