import pandas as pd
import numpy as np

data = pd.read_csv("final_predictions_lstm_glove1_all_folds.csv")
data.columns = ["index", "problem_log_id", "grader_teacher_id", "grade_1_prob_all", "grade_2_prob_all", "grade_3_prob_all", "grade_4_prob_all", "grade_5_prob_all" ]
data = data.drop(columns=["index"])
data = data.astype({'problem_log_id': 'int64'}, copy=True)
data = data.astype({'grader_teacher_id': 'int64'}, copy=True)

print(data)
data.to_csv("predictions_lstm_glove1_all_folds.csv")