import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import ast

ensemble_predictions = pd.read_csv("ensemble_predictions_all_3_1115.csv")
full_connected = pd.read_csv("vectorized_glove1.csv", converters={4: ast.literal_eval}, encoding="latin1")
full_connected = full_connected[["problem_log_id", "problem_id","grader_teacher_id", "encoded_grade", "folds"]]
merged = ensemble_predictions.merge(full_connected, on=["problem_log_id", "grader_teacher_id"])


problem_ids = set(merged["problem_id"])
auc_sample_lengths = {}
acc_sample_lengths = {}

id_auc = {}
id_accuracy = {}

for problem_id in problem_ids:
    problem_object = merged.loc[merged['problem_id'] == problem_id]


    actual = np.array(list(problem_object["encoded_grade"].values))
    predicted = np.array(list(problem_object[["grade_1_en", "grade_2_en", "grade_3_en", "grade_4_en", "grade_5_en"]].values))

    accuracy = accuracy_score(np.argmax(actual, axis=1), np.argmax(predicted, axis=1))
    id_accuracy[problem_id] = accuracy
    acc_sample_lengths[problem_id] = len(problem_object)

    # initialize as NaN in the case that only one class exists
    auc = np.nan
    try:
        auc = roc_auc_score(actual, predicted)
    except ValueError:
        pass

    if (np.isnan(auc)):
        pass
    else:
        id_auc[problem_id] = auc
        auc_sample_lengths[problem_id] = len(problem_object)

    # print("Problem:", problem_id, "AUC:", auc)
print("Number of problems:", len(id_auc))
sorted_auc = {k: v for k, v in sorted(id_auc.items(), key=lambda item: item[1], reverse=True)}
ids = list(map(str, list(sorted_auc.keys())))

y = np.arange(len(ids))

plt.bar(y, list(sorted_auc.values()), align='center', alpha=0.5)
plt.xticks(y, ids, rotation='vertical')
plt.ylabel('AUC')
plt.title('Problem Id vs AUC')

plt.show()

print("Number of problems:", len(id_accuracy))
sorted_acc = {k: v for k, v in sorted(id_accuracy.items(), key=lambda item: item[1], reverse=True)}
ids = list(map(str, list(sorted_acc.keys())))

y = np.arange(len(ids))

plt.bar(y, list(sorted_acc.values()), align='center', alpha=0.5)
plt.xticks(y, ids, rotation='vertical')
plt.ylabel('Accuracy')
plt.title('Problem Id vs Accuracy')

plt.show()
