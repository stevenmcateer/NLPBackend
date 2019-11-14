import import_data
import torch_lstm_all_folds
import RF
import torch
from torch.nn.utils.rnn import pad_sequence


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_fold_vectors(master_df):
    # LSTM data
    lstm_answers = []
    for ans in list(master_df["idx_answer"]):
        if ans == []:
            lstm_answers.append(torch.LongTensor([0]).to(device))
        else:
            if len(ans) > 50:
                print("ans is", len(ans))
                lstm_answers.append(torch.LongTensor(ans[:50]).to(device))
            else:
                lstm_answers.append(torch.LongTensor(ans).to(device))

    # RF data
    rf_answers = master_df["answer"]
    rf_grades = master_df["grade"]

    lstm_grades = list(master_df["encoded_grade"])
    all_folds = list(master_df["folds"])
    problem_log_ids = list(master_df["problem_log_id"])
    grader_teacher_ids = list(master_df["grader_teacher_id"])

    return lstm_answers, lstm_grades, rf_answers, rf_grades, problem_log_ids, grader_teacher_ids, all_folds


def run_trained_models(fold_num, lstm_train, rf_train):
    lstm_model = torch_lstm_all_folds.load_saved_model(fold_num)
    rf_model = RF.load_saved_model(fold_num)

    lstm_predictions = lstm_model(lstm_train)
    rf_predictions = rf_model(rf_train)

    return lstm_predictions, rf_predictions



def run_ensemble():
    dataset_name = "glove1"
    master_df, all_problems, vocab_list, weights_matrix = import_data.preprocessing(dataset_name)

    # Load answers and grades for each model
    lstm_answers, lstm_grades, rf_answers, rf_grades, problem_log_ids, grader_teacher_ids, list_folds = load_fold_vectors(master_df)

    seq_lengths = torch.LongTensor([len(ans) for ans in lstm_answers]).to(device)
    lstm_answers = pad_sequence(lstm_answers, batch_first=True, padding_value=0).to(device)
    lstm_grades = torch.tensor(lstm_grades, dtype=torch.long).to(device)

    for f in range(1, 11):
        # Slice up those LSTM indices
        lstm_test_index = [i for i, x in enumerate(list_folds) if x == f]
        for val in list_folds:
            if val != f:
                validation_fold_num = val
                break
        lstm_val_index = [i for i, x in enumerate(list_folds) if x == validation_fold_num]
        lstm_train_index = [i for i, x in enumerate(list_folds) if x != f and x != validation_fold_num]

        lstm_X_train, lstm_X_test = lstm_answers[lstm_train_index], lstm_answers[lstm_test_index]
        lstm_X_train_seqs, lstm_X_test_seqs, X_val_seqs = seq_lengths[lstm_train_index], seq_lengths[lstm_test_index], seq_lengths[lstm_val_index]
        lstm_y_train, lstm_y_test = lstm_grades[lstm_train_index], lstm_grades[lstm_test_index]
        X_validation, y_validation = lstm_answers[lstm_val_index], lstm_grades[lstm_val_index]

        # Reshape everything to have 3 dims
        # answers = answers.reshape(-1, answers.shape[1]).to(device)
        lstm_X_train = lstm_X_train.reshape(-1, lstm_answers.shape[1]).to(device)
        lstm_X_test = lstm_X_test.reshape(-1, lstm_answers.shape[1]).to(device)

        # RF indices
        rf_test_set = [i for i, x in enumerate(list_folds) if x == f]
        rf_training_set = [i for i, x in enumerate(list_folds) if x != f]

        print("Training fold:", f, "/", 10)

        # Run saved models with fold training data
        lstm_train_predictions, rf_train_predictions = run_trained_models(f, lstm_train, rf_train)