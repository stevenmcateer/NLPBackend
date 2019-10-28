import pandas as pd
import numpy as np

data = pd.read_csv("final_predictions_lstm_glove3.csv")
data.columns = ["index", "problem_log_id", "grader_teacher_id", "grade_1_prob", "grade_2_prob", "grade_3_prob", "grade_4_prob", "grade_5_prob" ]
data = data.drop(columns=["index"])
data = data.astype({'problem_log_id': 'int64'}, copy=True)
data = data.astype({'grader_teacher_id': 'int64'}, copy=True)

print(data)
data.to_csv("predictions_lstm_glove3.csv")