import import_data
import import_data_50

import torch_lstm_emb
import pandas as pd




if __name__ == "__main__":
    dataset_name = "anthony"

    # master_df, all_problems, vocab_list, weights_matrix = import_data.preprocessing(dataset_name)
    master_df, all_problems, vocab_list, weights_matrix = import_data_50.preprocessing(dataset_name)
    # print(master_df)

    # torch_model.train_test_all_problems(master_df, all_problems, dataset_name)
    # torch_lstm.train_test_all_problems(master_df, all_problems, dataset_name)
    torch_lstm_emb.train_test_all_problems(master_df, all_problems, dataset_name, vocab_list, weights_matrix)