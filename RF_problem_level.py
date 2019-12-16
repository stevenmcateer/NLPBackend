import pandas as pd
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
import evaluation




fully_connceted_data = pd.read_csv('full_connected_responses.csv', encoding='latin-1')
stanford_tokenized_data = pd.read_csv('stanford_tokenized_cleaned_answer.csv')
# n_words = pd.read_csv('nword_model.csv')
list(fully_connceted_data)

updated_fully_connected = pd.concat([fully_connceted_data, pd.get_dummies(fully_connceted_data.grade)], axis= 1)
updated_fully_connected = updated_fully_connected.drop(['grade_0', 'grade_1', 'grade_2', 'grade_3', 'grade_4'],  axis=1)
updated_fully_connected.rename(columns={1:'grade_0'}, inplace=True)
updated_fully_connected.rename(columns={2:'grade_1'}, inplace=True)
updated_fully_connected.rename(columns={3:'grade_2'}, inplace=True)
updated_fully_connected.rename(columns={4:'grade_3'}, inplace=True)
updated_fully_connected.rename(columns={5:'grade_4'}, inplace=True)
list(updated_fully_connected)

####MERGE THE DATA TO HAVE THE PARSED
tokenized_fully_connected = pd.merge(updated_fully_connected, stanford_tokenized_data, on = 'problem_log_id', how = 'left')



#train the count vectorizor on the whole corpus from the stanford tokenized answers
count_vectorizer = CountVectorizer(analyzer='word')  ####TreebankWordTokenizer tokenizer=TreebankWordTokenizer().tokenize,
all_word_counts = count_vectorizer.fit_transform(tokenized_fully_connected[~tokenized_fully_connected.parsed_cleaned_answers.isna()].parsed_cleaned_answers)

###calculate the tf-idf of the entire corpus
freq_word_transformer = TfidfTransformer()
all_freq_words_fitted = freq_word_transformer.fit_transform(all_word_counts)

no_missing_text = updated_fully_connected[~updated_fully_connected.cleaned_answer_text.isna()]
all_problem_ids = set(no_missing_text["problem_id"])


all_auc = []
all_rmse = []
all_final_predictions = []
test_order = []

count = 0
for problem_id in all_problem_ids:
# for problem_id in [1089733]:
    count+=1
    print("///////////////////////////////////// Problem:", count, "of", len(all_problem_ids))
    print("ID:", problem_id)

    problem_predictions = pd.DataFrame()
    true_predictions = pd.DataFrame()
    problem_log_ids = []
    grader_teacher_ids = []
    problem_auc = 0
    problem_rmse = 0

    # Create random forest model per problem
    rf_model = RandomForestClassifier(n_estimators=3,random_state=0, n_jobs=-1, criterion='gini')#criterion = 'gini' or 'entropy'

    problem_object = no_missing_text.loc[no_missing_text['problem_id'] == problem_id]
    problem_id_list = problem_object["problem_id"]

    labels = problem_object[['grade_0', 'grade_1', 'grade_2', 'grade_3', 'grade_4']].values

    unique_problem_folds = np.unique(problem_object.folds.values)
    for i in unique_problem_folds:
        print("Fold:", i)
        print("unique folds:", unique_problem_folds)

        # if the problem only has 1 fold
        if len(unique_problem_folds) == 1:
            for i in range(len(labels)):
                row = pd.DataFrame({"0": [0], "1": [0], "2": [0], "3": [0], "4": [1]})
                problem_predictions = problem_predictions.append(row)
                true_predictions = labels
                problem_log_ids = problem_object["problem_log_id"].values
                grader_teacher_ids = problem_object["grader_teacher_id"].values

            break

        test_set = problem_object.loc[problem_object.folds == i]
        training_set = problem_object.loc[problem_object.folds != i]

        np.unique(test_set.folds)
        np.unique(training_set.folds)

        X_train = training_set.cleaned_answer_text.values
        # print("X_train\n", X_train)
        # print(len(X_train))
        X_test = test_set.cleaned_answer_text.values
        y_train = training_set[['grade_0', 'grade_1', 'grade_2', 'grade_3', 'grade_4']].values
        # print("y_train\n", y_train)
        y_test = test_set[['grade_0', 'grade_1', 'grade_2', 'grade_3', 'grade_4']].values

        # keep track of order for these
        problem_log_id_test = test_set["problem_log_id"].values
        grader_teacher_id_test = test_set["grader_teacher_id"].values

        problem_log_ids.extend(problem_log_id_test)
        grader_teacher_ids.extend(grader_teacher_id_test)


        ##########transform the training data to the counting vector (matches the counts to our full corpus)
        X_train_wc = count_vectorizer.transform(X_train)
        X_train_freq_words = freq_word_transformer.transform(X_train_wc)
        # print("X_freq_words:", X_train_freq_words.shape)
        # print("X_train_len:", len(X_train))
        # print("y_train_len:", len(y_train))

        ########## transform the test data
        X_test_wc = count_vectorizer.transform(X_test)  ###this is transforming
        X_test_freq_words = freq_word_transformer.transform(X_test_wc)

        ########## FIT THE RANDOM FOREST
        # print("fitting this shit with:")
        # print("X_train_freq_words\n", X_train_freq_words, "\n")
        # print("y_train\n", y_train)
        random_forest = rf_model.fit(X_train_freq_words, pd.DataFrame(y_train))

        rf_predict_traditional = random_forest.predict_proba(X_test_freq_words)

        if len(rf_predict_traditional[0][0]) == 1 or \
            len(rf_predict_traditional[1][0]) == 1:
            for i in range(len(y_test)):
                row = pd.DataFrame({"yes_0": [0], "yes_1": [0], "yes_2": [0], "yes_3": [0], "yes_4":[1]})
                problem_predictions = problem_predictions.append(row)
        elif len(rf_predict_traditional[2][0]) == 1 or \
            len(rf_predict_traditional[3][0]) == 1 or \
            len(rf_predict_traditional[4][0]) == 1:
            for i in range(len(y_test)):
                row = pd.DataFrame({"yes_0": [1], "yes_1": [0], "yes_2": [0], "yes_3": [0], "yes_4":[0]})
                problem_predictions = problem_predictions.append(row)
        else:
            grade_0 = pd.DataFrame(rf_predict_traditional[0])
            grade_0.rename(columns={0:'no_0'}, inplace=True)
            grade_0.rename(columns={1: 'yes_0'}, inplace=True)
            grade_1 = pd.DataFrame(rf_predict_traditional[1])
            grade_1.rename(columns={0: 'no_1'}, inplace=True)
            grade_1.rename(columns={1: 'yes_1'}, inplace=True)
            grade_2 = pd.DataFrame(rf_predict_traditional[2])
            grade_2.rename(columns={0: 'no_2'}, inplace=True)
            grade_2.rename(columns={1: 'yes_2'}, inplace=True)
            grade_3 = pd.DataFrame(rf_predict_traditional[3])
            grade_3.rename(columns={0: 'no_3'}, inplace=True)
            grade_3.rename(columns={1: 'yes_3'}, inplace=True)
            grade_4 = pd.DataFrame(rf_predict_traditional[4])
            grade_4.rename(columns={0: 'no_4'}, inplace=True)
            grade_4.rename(columns={1: 'yes_4'}, inplace=True)

            all_probabilities = pd.concat([grade_0, grade_1], axis=1)
            all_probabilities = pd.concat([all_probabilities, grade_2], axis=1)
            all_probabilities = pd.concat([all_probabilities, grade_3], axis=1)
            all_probabilities = pd.concat([all_probabilities, grade_4], axis=1)
            print(all_probabilities)
            predicted_grade = pd.concat([grade_0['yes_0'], grade_1['yes_1']], axis=1)
            predicted_grade = pd.concat([predicted_grade, grade_2['yes_2']], axis=1)
            predicted_grade = pd.concat([predicted_grade, grade_3['yes_3']], axis=1)
            predicted_grade = pd.concat([predicted_grade, grade_4['yes_4']], axis=1)
            problem_predictions = problem_predictions.append(predicted_grade)

        true_predictions = true_predictions.append(pd.DataFrame(y_test))


    print("y_test", true_predictions)
    print("preds", problem_predictions)

    # initialize as NaN in the case that only one class exists
    auc = np.nan
    try:
        auc = roc_auc_score(np.array(true_predictions), np.array(problem_predictions))
        # auc = evaluation.auc(np.array(true_predictions), np.array(problem_predictions))
    except ValueError:
        pass

    if (np.isnan(auc)):
        print("AUC is nan, not appending anything")
    else:
        all_auc.append(auc)

    # RMSE
    mean_sq_error_rf = mean_squared_error(np.array(true_predictions), np.array(problem_predictions))
    rmse = math.sqrt(mean_sq_error_rf)
    all_rmse.append(rmse)

    saved_predictions = pd.DataFrame(problem_predictions)
    # print("Saved preds\n")
    # print(saved_predictions)
    saved_predictions['problem_log_id'] = problem_log_ids
    saved_predictions['grader_teacher_id'] = grader_teacher_ids
    saved_predictions['problem_id'] = problem_id_list
    all_final_predictions.append(np.array(saved_predictions))
    print("len of preds so far", len(all_final_predictions))


    print("AUC:", auc)
    print("RMSE:", rmse)

random_forest_predictions = pd.DataFrame(np.concatenate(all_final_predictions))


random_forest_predictions[5] = random_forest_predictions[5].astype('int')
random_forest_predictions.rename(columns={0: 'grade_1_rf_prob'}, inplace=True)
random_forest_predictions.rename(columns={1: 'grade_2_rf_prob'}, inplace=True)
random_forest_predictions.rename(columns={2: 'grade_3_rf_prob'}, inplace=True)
random_forest_predictions.rename(columns={3: 'grade_4_rf_prob'}, inplace=True)
random_forest_predictions.rename(columns={4: 'grade_5_rf_prob'}, inplace=True)
random_forest_predictions.rename(columns={5: 'problem_log_id'}, inplace=True)
random_forest_predictions.rename(columns={6: 'grader_teacher_id'}, inplace=True)
random_forest_predictions.rename(columns={7: 'problem_id'}, inplace=True)
random_forest_predictions['grader_teacher_id']=random_forest_predictions['grader_teacher_id'].astype('int')

print("Num of AUC values:", len(all_auc))
average_auc_folds = np.mean(all_auc)
print('10-fold AUC:', average_auc_folds )####
average_rmse_folds = np.mean(all_rmse) ###
print('10-fold RMSE:', average_rmse_folds)


random_forest_predictions.to_csv('rf_predictions_prob_level.csv', index = False)
