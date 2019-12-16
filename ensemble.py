import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
import evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
import math

# load the datasets
lstm_preds = pd.read_csv("predictions_lstm_glove1.csv")
lstm_all = pd.read_csv("predictions_lstm_glove1_all_folds.csv")
rf_preds = pd.read_csv("rf_predictions_fold.csv")
rf_prob_level = pd.read_csv("rf_predictions_prob_level.csv")

full_connected = pd.read_csv("vectorized_glove1.csv", converters={4: ast.literal_eval}, encoding="latin1")
full_connected = full_connected[["problem_log_id", "grader_teacher_id", "encoded_grade", "folds"]]

merged_preds = lstm_preds.merge(rf_preds, on=["problem_log_id", "grader_teacher_id"])
all_preds = merged_preds.merge(lstm_all, on=["problem_log_id", "grader_teacher_id"])
final_preds = all_preds.merge(rf_prob_level, on=["problem_log_id", "grader_teacher_id"])
print(final_preds)

merged_w_labels = final_preds.merge(full_connected, on=["problem_log_id", "grader_teacher_id"])
complete_X_y = merged_w_labels[["problem_log_id", "grader_teacher_id","folds", "grade_1_prob", "grade_2_prob", "grade_3_prob", "grade_4_prob", "grade_5_prob",
                       "grade_1_prob_all", "grade_2_prob_all", "grade_3_prob_all", "grade_4_prob_all", "grade_5_prob_all",
                       "grade_1_rf", "grade_2_rf", "grade_3_rf", "grade_4_rf", "grade_5_rf",
                       "grade_1_rf_prob", "grade_2_rf_prob", "grade_3_rf_prob", "grade_4_rf_prob", "grade_5_rf_prob", "encoded_grade"]]

features = complete_X_y.columns[3:23]
print("Features:", features)


all_folds = complete_X_y.folds.values
unique_folds =  np.unique(all_folds)

rf_model = RandomForestClassifier(n_estimators=100,random_state=0, n_jobs=-1)#criterion = 'gini' or 'entropy'
all_probs = pd.DataFrame()
all_auc = []
all_rmse = []
list_predictions = []

for i in unique_folds:
    print("Fold:", i)
    test_set = complete_X_y.loc[complete_X_y.folds == i]
    training_set = complete_X_y.loc[complete_X_y.folds != i]

    X_test = test_set[["grade_1_prob", "grade_2_prob", "grade_3_prob", "grade_4_prob", "grade_5_prob",
                      "grade_1_prob_all", "grade_2_prob_all", "grade_3_prob_all", "grade_4_prob_all", "grade_5_prob_all",
                      "grade_1_rf", "grade_2_rf", "grade_3_rf", "grade_4_rf", "grade_5_rf",
                       "grade_1_rf_prob", "grade_2_rf_prob", "grade_3_rf_prob", "grade_4_rf_prob", "grade_5_rf_prob"]].values
    y_test = list(test_set["encoded_grade"].values)

    X_train = training_set[["grade_1_prob", "grade_2_prob", "grade_3_prob", "grade_4_prob", "grade_5_prob",
                      "grade_1_prob_all", "grade_2_prob_all", "grade_3_prob_all", "grade_4_prob_all", "grade_5_prob_all",
                      "grade_1_rf", "grade_2_rf", "grade_3_rf", "grade_4_rf", "grade_5_rf",
                      "grade_1_rf_prob", "grade_2_rf_prob", "grade_3_rf_prob", "grade_4_rf_prob", "grade_5_rf_prob"]].values
    y_train = list(training_set["encoded_grade"].values)


    fitted_model = rf_model.fit(X_train, y_train)

    y_predict = fitted_model.predict_proba(X_test)
    # print("Shape of predict:", len(y_predict), len(y_predict[0]))

    # print(f'Model Accuracy: {rf_model.score(, y)}')

    # grade_0_preds = list(y_predict[0])
    # grade_1_preds = list(y_predict[1])
    # grade_2_preds = list(y_predict[2])
    # grade_3_preds = list(y_predict[3])
    # grade_4_preds = list(y_predict[4])
    # data = {"grade_1_pred": grade_0_preds,
    #         "grade_2_pred": grade_1_preds,
    #         "grade_3_pred": grade_2_preds,
    #         "grade_4_pred": grade_3_preds,
    #         "grade_5_pred": grade_4_preds}
    # data_df = pd.DataFrame(data)
    # all_probs = pd.concat([data_df, all_probs])
    #
    grade_0 = pd.DataFrame(y_predict[0])
    grade_0.rename(columns={0: 'no_0'}, inplace=True)
    grade_0.rename(columns={1: 'yes_0'}, inplace=True)
    grade_1 = pd.DataFrame(y_predict[1])
    grade_1.rename(columns={0: 'no_1'}, inplace=True)
    grade_1.rename(columns={1: 'yes_1'}, inplace=True)
    grade_2 = pd.DataFrame(y_predict[2])
    grade_2.rename(columns={0: 'no_2'}, inplace=True)
    grade_2.rename(columns={1: 'yes_2'}, inplace=True)
    grade_3 = pd.DataFrame(y_predict[3])
    grade_3.rename(columns={0: 'no_3'}, inplace=True)
    grade_3.rename(columns={1: 'yes_3'}, inplace=True)
    grade_4 = pd.DataFrame(y_predict[4])
    grade_4.rename(columns={0: 'no_4'}, inplace=True)
    grade_4.rename(columns={1: 'yes_4'}, inplace=True)

    all_probabilities = pd.concat([grade_0, grade_1], axis=1)
    all_probabilities = pd.concat([all_probabilities, grade_2], axis=1)
    all_probabilities = pd.concat([all_probabilities, grade_3], axis=1)
    all_probabilities = pd.concat([all_probabilities, grade_4], axis=1)

    predicted_grade = pd.concat([grade_0['yes_0'], grade_1['yes_1']], axis=1)
    predicted_grade = pd.concat([predicted_grade, grade_2['yes_2']], axis=1)
    predicted_grade = pd.concat([predicted_grade, grade_3['yes_3']], axis=1)
    predicted_grade = pd.concat([predicted_grade, grade_4['yes_4']], axis=1)


    y_test_dataframe = pd.DataFrame(y_test)

    mean_sq_error_rf = mean_squared_error(y_test, np.array(predicted_grade))
    rmse_rf = math.sqrt(mean_sq_error_rf)

    auc_rf = roc_auc_score(np.array(y_test), np.array(predicted_grade))
    print("Fold AUC:", auc_rf)
    print("Fold RMSE:", rmse_rf)
    all_auc.append(auc_rf)
    all_rmse.append(rmse_rf)

    saved_predictions = predicted_grade
    saved_predictions['problem_log_id'] = test_set['problem_log_id'].values
    saved_predictions['grader_teacher_id'] = test_set['grader_teacher_id'].values
    list_predictions.append(np.array(saved_predictions))

random_forest_predictions = pd.DataFrame(np.concatenate(list_predictions))

random_forest_predictions[5] = random_forest_predictions[5].astype('int')
random_forest_predictions.rename(columns={0: 'grade_1_en'}, inplace=True)
random_forest_predictions.rename(columns={1: 'grade_2_en'}, inplace=True)
random_forest_predictions.rename(columns={2: 'grade_3_en'}, inplace=True)
random_forest_predictions.rename(columns={3: 'grade_4_en'}, inplace=True)
random_forest_predictions.rename(columns={4: 'grade_5_en'}, inplace=True)
random_forest_predictions.rename(columns={5: 'problem_log_id'}, inplace=True)
random_forest_predictions.rename(columns={6: 'grader_teacher_id'}, inplace=True)
random_forest_predictions['grader_teacher_id']=random_forest_predictions['grader_teacher_id'].astype('int')

random_forest_predictions.to_csv('ensemble_predictions_all_4_1124.csv', index = False)

final_auc = np.mean(all_auc)
print("Final AUC:", final_auc)
final_rmse = np.mean(all_rmse)
print("Final RMSE:", final_rmse)

importances = rf_model.feature_importances_
print(list(zip(features, importances)))
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation='vertical', color='r', edgecolor='k', linewidth=1.2)
# Tick labels for x axis
plt.xticks(x_values, features, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance')
plt.xlabel('Variable')
plt.title('Variable Importances')
plt.show()

