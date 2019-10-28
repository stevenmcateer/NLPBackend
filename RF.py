import pandas as pd
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score





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

tokenized_fully_connected[['cleaned_answer_text', 'parsed_cleaned_answers']]



#train the count vectorizor on the whole corpus from the stanford tokenized answers
counting_tool_traditional = CountVectorizer(analyzer='word')  ####TreebankWordTokenizer tokenizer=TreebankWordTokenizer().tokenize,
counting_words_traditional = counting_tool_traditional.fit_transform(tokenized_fully_connected[~tokenized_fully_connected.parsed_cleaned_answers.isna()].parsed_cleaned_answers)

# counting_words_traditional


###calculate the tf-idf of the entire corpus
term_freq_tool_traditional = TfidfTransformer()
term_freq_words_traditional = term_freq_tool_traditional.fit_transform(counting_words_traditional)


###### SET FOLDER FOR AUC's
all_auc = []
all_rmse = []
list_predictions = []



#random forest model
rf_model = RandomForestClassifier(n_estimators=100,random_state=0, n_jobs=-1)#criterion = 'gini' or 'entropy'


#no missing text
no_missing_text = updated_fully_connected[~updated_fully_connected.cleaned_answer_text.isna()]

for i in np.unique(no_missing_text.folds.values):
    print(i)
    test_set = no_missing_text.loc[no_missing_text.folds == i]
    training_set = no_missing_text.loc[no_missing_text.folds != i]
    # training_set = no_missing_text.loc[~(no_missing_text.folds == i)]#does the same thing

    np.unique(test_set.folds)
    np.unique(training_set.folds)

    X_train = training_set.cleaned_answer_text.values
    X_test = test_set.cleaned_answer_text.values
    y_train = training_set[['grade_0', 'grade_1', 'grade_2', 'grade_3', 'grade_4']].values
    y_test = test_set[['grade_0', 'grade_1', 'grade_2', 'grade_3', 'grade_4']].values

    ##########transform the training data to the counting vector (matches the counts to our full corpus)
    counting_words_training_traditional = counting_tool_traditional.transform(X_train)###this is transforming
    counting_words_training_traditional
    term_freq_words_transform_training_traditional = term_freq_tool_traditional.transform(counting_words_training_traditional)
    ########## transform the test data
    counting_words_training_traditional_test = counting_tool_traditional.transform(X_test)  ###this is transforming
    term_freq_words_transform_training_traditional_test = term_freq_tool_traditional.transform(counting_words_training_traditional_test)




    ########## FIT THE RANDOM FOREST
    random_forest = rf_model.fit(term_freq_words_transform_training_traditional, pd.DataFrame(y_train))

    rf_predict_traditional = random_forest.predict_proba(term_freq_words_transform_training_traditional_test)

    grade_0 = pd.DataFrame(rf_predict_traditional[0])
    # grade_0 = pd.DataFrame(rf_predict_traditional_ovr[0])
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


    predicted_grade = pd.concat([grade_0['yes_0'], grade_1['yes_1']], axis=1)
    predicted_grade = pd.concat([predicted_grade, grade_2['yes_2']], axis=1)
    predicted_grade = pd.concat([predicted_grade, grade_3['yes_3']], axis=1)
    predicted_grade = pd.concat([predicted_grade, grade_4['yes_4']], axis=1)
    np.array(predicted_grade)
    y_test_dataframe = pd.DataFrame(y_test)



    mean_sq_error_rf = mean_squared_error(y_test, np.array(predicted_grade))
    RMSE_rf= math.sqrt(mean_sq_error_rf)

    auc_rf = roc_auc_score(y_test,  np.array(predicted_grade))
    all_auc.append(auc_rf)
    all_rmse.append(RMSE_rf)


    saved_predictions = predicted_grade
    saved_predictions['problem_log_id'] = test_set['problem_log_id'].values
    saved_predictions['grader_teacher_id'] = test_set['grader_teacher_id'].values
    list_predictions.append(np.array(saved_predictions))





random_forest_predictions = pd.DataFrame(np.concatenate(list_predictions))


random_forest_predictions[5] = random_forest_predictions[5].astype('int')
random_forest_predictions.rename(columns={0: 'grade_1_rf'}, inplace=True)
random_forest_predictions.rename(columns={1: 'grade_2_rf'}, inplace=True)
random_forest_predictions.rename(columns={2: 'grade_3_rf'}, inplace=True)
random_forest_predictions.rename(columns={3: 'grade_4_rf'}, inplace=True)
random_forest_predictions.rename(columns={4: 'grade_5_rf'}, inplace=True)
random_forest_predictions.rename(columns={5: 'problem_log_id'}, inplace=True)
random_forest_predictions.rename(columns={6: 'grader_teacher_id'}, inplace=True)
random_forest_predictions['grader_teacher_id']=random_forest_predictions['grader_teacher_id'].astype('int')

average_auc_folds = np.mean(all_auc)
print('10-fold AUC : ', average_auc_folds )####
average_rmse_folds = np.mean(all_rmse) ###
print('10-fold RMSE : ,', average_rmse_folds)


random_forest_predictions.to_csv('random_forest_predictions_fold_STEVE_example.csv', index = False)
