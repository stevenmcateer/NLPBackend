import import_data
import torch_lstm_all_folds
import torch_lstm_folds




if __name__ == "__main__":
    dataset_name = "glove1"

    master_df, all_problems, vocab_list, weights_matrix = import_data.preprocessing(dataset_name)

    # torch_lstm_folds.train_test_all_problems(master_df, all_problems, dataset_name, vocab_list, weights_matrix)
    torch_lstm_all_folds.train_test_all_folds(master_df, dataset_name, vocab_list, weights_matrix)
