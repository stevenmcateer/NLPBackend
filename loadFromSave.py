import matplotlib
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import os
import pandas as pd
pd.set_option('display.max_columns', 10)
from sklearn.metrics import accuracy_score
import math
from sklearn import tree
from sklearn.preprocessing import label_binarize
import time
import numpy as np
from sklearn.metrics import f1_score, r2_score
from sklearn.metrics import mean_squared_error
from skll.metrics import kappa as kpa
from sklearn.externals import joblib
start_time = time.time()
def auc(actual, predicted, average_over_labels=True, partition=1024.):
    assert len(actual) == len(predicted)

    ac = np.array(actual, dtype=np.float32).reshape((len(actual),-1))
    pr = np.array(predicted, dtype=np.float32).reshape((len(predicted),-1))

    na = np.argwhere([not np.any(np.isnan(i)) for i in ac]).ravel()

    ac = ac[na]
    pr = pr[na]

    label_auc = []
    for i in range(ac.shape[-1]):
        a = np.array(ac[:,i])
        p = np.array(pr[:,i])

        val = np.unique(a)

        # if len(val) > 2:
        #     print('AUC Warning - Number of distinct values in label set {} is greater than 2, '
        #           'using median split of distinct values...'.format(i))
        if len(val) == 1:
            # print('AUC Warning - There is only 1 distinct value in label set {}, unable to calculate AUC'.format(i))
            label_auc.append(np.nan)
            continue

        pos = np.argwhere(a[:] >= np.median(val))
        neg = np.argwhere(a[:] < np.median(val))

        # print(pos)
        # print(neg)

        p_div = int(np.ceil(len(pos)/partition))
        n_div = int(np.ceil(len(neg)/partition))

        # print(len(pos), p_div)
        # print(len(neg), n_div)

        div = 0
        for j in range(int(p_div)):
            p_range = list(range(int(j * partition), int(np.minimum(int((j + 1) * partition), len(pos)))))
            for k in range(n_div):
                n_range = list(range(int(k * partition), int(np.minimum(int((k + 1) * partition), len(neg)))))


                eq = np.ones((np.alen(neg[n_range]), np.alen(pos[p_range]))) * p[pos[p_range]].T == np.ones(
                    (np.alen(neg[n_range]), np.alen(pos[p_range]))) * p[neg[n_range]]

                geq = np.array(np.ones((np.alen(neg[n_range]), np.alen(pos[p_range]))) *
                               p[pos[p_range]].T >= np.ones((np.alen(neg[n_range]),
                                                             np.alen(pos[p_range]))) * p[neg[n_range]],
                               dtype=np.float32)
                geq[eq[:, :] == True] = 0.5

                # print(geq)
                div += np.sum(geq)
                # print(np.sum(geq))
                # exit(1)

        label_auc.append(div / (np.alen(pos)*np.alen(neg)))
        # print(label_auc)

    if average_over_labels:
        return np.nanmean(label_auc)
    else:
        return label_auc


def f1(actual, predicted):
    return f1_score(np.array(actual), np.round(predicted))


def rmse(actual, predicted, average_over_labels=True):
    assert len(actual) == len(predicted)

    ac = np.array(actual, dtype=np.float32).reshape((len(actual), -1))
    pr = np.array(predicted, dtype=np.float32).reshape((len(predicted), -1))

    na = np.argwhere([not np.any(np.isnan(i)) for i in ac]).ravel()

    if len(na) == 0:
        return np.nan

    ac = ac[na]
    pr = pr[na]

    label_rmse = []
    for i in range(ac.shape[-1]):
        dif = np.array(ac[:, i]) - np.array(pr[:, i])
        sqdif = dif**2
        mse = np.nanmean(sqdif)
        label_rmse.append(np.sqrt(mse))


    if average_over_labels:
        return np.nanmean(label_rmse)
    else:
        return label_rmse


def cohen_kappa(actual, predicted, split=0.5, average_over_labels=True):
    assert len(actual) == len(predicted)

    ac = np.array(actual,dtype=np.float32).reshape((len(actual), -1))
    pr = np.array(predicted,dtype=np.float32).reshape((len(predicted), -1))

    na = np.argwhere([not np.any(np.isnan(i)) for i in ac]).ravel()

    if len(na) == 0:
        return np.nan

    ac = np.array(np.array(ac[na]) > split, dtype=np.int32)
    pr = np.array(np.array(pr[na]) > split, dtype=np.int32)

    label_kpa = []
    if hasattr(split, '__iter__'):
        assert len(split) == ac.shape[-1]
    else:
        split = np.ones(ac.shape[1]) * split

    for i in range(ac.shape[-1]):
        label_kpa.append(kpa(np.array(np.array(ac[:, i]) > split[i], dtype=np.int32),
                np.array(np.array(pr[:, i]) > split[i], dtype=np.int32)))

    if average_over_labels:
        return np.nanmean(label_kpa)
    else:
        return label_kpa


def cohen_kappa_multiclass(actual, predicted):
    assert len(actual) == len(predicted)

    ac = np.array(actual,dtype=np.float32).reshape((len(actual), -1))
    pr = np.array(predicted,dtype=np.float32).reshape((len(predicted), -1))

    try:
        na = np.argwhere([not np.any(np.isnan(i)) for i in ac]).ravel()
    except:
        for i in ac:
            print(i)

        for i in ac:
            print(np.any(np.isnan(i)))

    if len(na) == 0:
        return np.nan

    aci = np.argmax(np.array(np.array(ac[na]), dtype=np.int32), axis=1)
    pri = np.argmax(np.array(np.array(pr[na]), dtype=np.float32), axis=1)

    # for i in range(len(aci)):
    #     print(aci[i],'--',pri[i],':',np.array(pr[na])[i])

    return kpa(aci,pri)

# def kappa(actual, predicted, split=0.5):
#     # pred = normalize(list(predicted), method='uniform')
#     return kpa(actual, [p > split for p in predicted])
# os.chdir('/Users/johnerickson/Desktop/Assistments_Research/Open_Response')
# orig_engage_ny = pd.read_csv('EXTREME_FINAL_1112pm.csv')
engage_ny = pd.read_csv('open_response_filter.csv')
y = label_binarize(np.array(engage_ny.correct).astype('str'), classes=['0.0', '0.25', '0.5', '0.75', '1.0'])
y.shape
matplotlib.rcParams.update({'errorbar.capsize': 6})
practice = pd.concat([engage_ny.drop('correct', axis=1),
                      pd.get_dummies(engage_ny['correct'])], axis=1)
practice = practice.groupby('problem_id').sum().reset_index()
check_amount_responses = engage_ny.loc[engage_ny['problem_id'] == 1276708]
check_amount_responses.shape
problems_with_only_grade_1_given = practice.loc[(practice[0.0] == 0) &
                                                (practice[0.25] == 0) &
                                                (practice[0.5] == 0) &
                                                (practice[0.75] == 0)]
problems_with_only_grade_0_given = practice.loc[(practice[1.0] == 0) &
                                                (practice[0.25] == 0) &
                                                (practice[0.5] == 0) &
                                                (practice[0.75] == 0)]
dummy_predictors = pd.get_dummies(engage_ny['correct'])
dummy_predictors.index.values
#################### Traditional LOOCV
accuracy_overall = []
final_pred_prob_list = pd.DataFrame([])
anthony_rmse_overall = []
anthony_overall_kappa = []
anthony_overall_auc_prob = []
anthony_overall_auc = []
list_problems_individual_traditional = engage_ny.groupby('problem_id').count().reset_index()  ## THere are 113 problems
unique_ids_traditional = list_problems_individual_traditional['problem_id']
predictions_for_accuracy =[]
test_actual_values_full_data = []
predictions_traditional = []
prediction_for_saving = []
RMSE_All_traditional = []
RMSE_All_from_columns_traditional = []
shape_traditional = []
for l in unique_ids_traditional:
    engage_ny_uniq_id_traditional = engage_ny.loc[engage_ny['problem_id'] == l]
    engage_ny_uniq_id_traditional.shape
    predictions_per_loop_traditional = []
    test_actual_values_each_loop_traditional = []
    test_actual_values_each_loop_traditional2 = []
    RMSE_each_loop_traditional = []

    for r in range(len(engage_ny_uniq_id_traditional)):
        test_set_traditional = engage_ny_uniq_id_traditional.loc[
            engage_ny_uniq_id_traditional.index.values == engage_ny_uniq_id_traditional.index.values[r]]
        test_y_value_traditional = test_set_traditional.correct.astype('str')
        dimensions_traditional = engage_ny_uniq_id_traditional.drop(test_set_traditional.index).shape[0]
        training_set_traditional = engage_ny_uniq_id_traditional.drop(test_set_traditional.index)
        training_set_traditional.shape
        test_actual_values_each_loop_traditional2.append(test_y_value_traditional)
        answers_traditional = training_set_traditional.answer_text.astype('str')
        correct_orig_traditional = training_set_traditional.correct.astype('str')
        correct_orig_traditional.shape
        correct_traditional = pd.merge(pd.DataFrame(correct_orig_traditional), dummy_predictors, left_index=True,
                                       right_index=True)
        correct_traditional.shape
        correct_traditional = correct_traditional[[0.0, 0.25, 0.5, 0.75, 1.0]].astype('str')
        text_x_value_traditional = test_set_traditional.answer_text.astype('str')
        ###coun     number of times each word occurs
        counting_tool_traditional = CountVectorizer(tokenizer=TreebankWordTokenizer().tokenize,
                                                    analyzer='word')  ####TreebankWordTokenizer?
        counting_words_traditional = counting_tool_traditional.fit_transform(answers_traditional)  # .split('\n')
        counting_words_test_traditional = counting_tool_traditional.transform(text_x_value_traditional)
        counting_words_traditional
        term_freq_tool_traditional = TfidfTransformer()
        term_freq_words_traditional = term_freq_tool_traditional.fit_transform(counting_words_traditional)
        term_freq_words_test_traditional = term_freq_tool_traditional.transform(counting_words_test_traditional)
        #decision_tree_traditional = tree.DecisionTreeClassifier(max_depth=3)#, max_features='log2')#, class_weight='balanced')

        # dt_fit_traditional = decision_tree_traditional.fit(term_freq_words_traditional, correct_traditional)
        filename = '.\Models\model-'+str(l)+'.sav'
        dt_fit_traditional = joblib.load(filename)
        tree_predict_probability_traditional = dt_fit_traditional.predict_proba(
            term_freq_words_test_traditional)  ###predict the probabilities
        correct_traditional.shape


        tree_predict_traditional = dt_fit_traditional.predict(
            term_freq_words_test_traditional)
        tree_predict_traditional_pd = pd.DataFrame(tree_predict_traditional)
        tree_predict_traditional_pd['problem_id']=engage_ny_uniq_id_traditional.problem_id.unique()
        tree_predict_traditional_pd = tree_predict_traditional_pd.values.tolist()
        predictions_for_accuracy.append(tree_predict_traditional_pd)
        correct_test_traditional_acc = pd.merge(pd.DataFrame(test_set_traditional), dummy_predictors, left_index=True,
                                            right_index=True)
        correct_test_traditional_acc = correct_test_traditional_acc.drop(['teacher_id', 'assignment_id', 'Unnamed: 0',
                                                                  'user_id', 'assistment_id', 'problem_text', 'correct',
                                                                  'answer_text', 'teacher_comment', 'problem_id'], axis=1)
        correct_test_traditional_acc = np.array(correct_test_traditional_acc).astype(int)
        tree_predict_traditional.astype(int)#predicted
        correct_test_traditional_acc#actual
        accuracy_percent = accuracy_score(correct_test_traditional_acc, tree_predict_traditional)# calculate the prediction
        accuracy_overall.append(accuracy_percent)
        correct_test_traditional = pd.merge(pd.DataFrame(test_set_traditional), dummy_predictors, left_index=True,
                                            right_index=True)
        anthony = auc(correct_test_traditional_acc[0], tree_predict_traditional[0], average_over_labels=True,
                      partition=1024.)
        anthony_overall_auc.append(anthony)
        loop_prob_auc = tree_predict_probability_traditional
        loop_prob_auc = pd.DataFrame.from_records(loop_prob_auc)
        loop_prob_auc = loop_prob_auc[0].str[0]
        loop_prob_auc = 1 - loop_prob_auc
        loop_prob_auc = np.array(loop_prob_auc)
        anthony_prob = auc(correct_test_traditional_acc[0], loop_prob_auc, average_over_labels=True,
                           partition=1024.)
        anthony_overall_auc_prob.append(anthony_prob)
        correct_test_traditional = correct_test_traditional.drop(['teacher_id', 'assignment_id', 'Unnamed: 0',
                                                                  'user_id', 'assistment_id', 'problem_text', 'correct',
                                                                  'answer_text', 'teacher_comment'], axis=1)

        correct_test_traditional = np.array(correct_test_traditional)

        test_actual_values_each_loop_traditional.append(correct_test_traditional)
        test_actual_values_full_data.append(correct_test_traditional)
        tree_predict_probability_traditional.extend([np.array(engage_ny_uniq_id_traditional.problem_id.unique())])
        predictions_traditional.append(tree_predict_probability_traditional)
        predictions_per_loop_traditional.append(tree_predict_probability_traditional)
        prediction_dataframe_traditional = pd.DataFrame(predictions_per_loop_traditional).reset_index()
        prediction_dataframe_traditional[5] = prediction_dataframe_traditional[5].str[0]
        prediction_dataframe_traditional[4] = prediction_dataframe_traditional[4].str[0]
        prediction_dataframe_traditional[4] = 1 - (prediction_dataframe_traditional[4].str[0])
        prediction_dataframe_traditional[3] = prediction_dataframe_traditional[3].str[0]
        prediction_dataframe_traditional[3] = 1 - (prediction_dataframe_traditional[3].str[0])
        prediction_dataframe_traditional[2] = prediction_dataframe_traditional[2].str[0]
        prediction_dataframe_traditional[2] = 1 - (prediction_dataframe_traditional[2].str[0])
        prediction_dataframe_traditional[1] = prediction_dataframe_traditional[1].str[0]
        prediction_dataframe_traditional[1] = 1 - (prediction_dataframe_traditional[1].str[0])
        prediction_dataframe_traditional[0] = prediction_dataframe_traditional[0].str[0]
        prediction_dataframe_traditional[0] = 1 - (prediction_dataframe_traditional[0].str[0])
        prediction_dataframe_traditional.columns = ['index', '0.0_predicted_prob', '0.25_predicted_prob',
                                                    '0.5_predicted_prob', '0.75_predicted_prob', '1.0_predicted_prob',
                                                    'problem_id']
        test_values_dataframe_traditional = pd.DataFrame.from_records(test_actual_values_each_loop_traditional)
        test_values_dataframe_traditional = pd.DataFrame(test_values_dataframe_traditional[0].values.tolist(),
                                                         columns=['problem_id_test', '0.0', '0.25', '0.5', '0.75',
                                                                  '1.0'])
        final_predictions_traditional = pd.concat([prediction_dataframe_traditional, test_values_dataframe_traditional],
                                                  axis=1)
        list(final_predictions_traditional)
        final_pred_prob_list.append(final_predictions_traditional)        
       
    list(final_predictions_traditional)
    final_predictions_traditional = final_predictions_traditional.drop(['problem_id_test'], axis=1)
    final_predictions_auc = final_predictions_traditional[
        ['0.0_predicted_prob', '0.25_predicted_prob', '0.5_predicted_prob', '0.75_predicted_prob',
         '1.0_predicted_prob']]
    final_predictions_auc = np.array(final_predictions_auc)
    final_predictions_auc_correct = final_predictions_traditional[['0.0', '0.25', '0.5', '0.75', '1.0']]
    final_predictions_auc_correct = np.array(final_predictions_auc_correct)
    anthony = auc(final_predictions_auc_correct, final_predictions_auc, average_over_labels=True,
                  partition=1024.)
    anthony_overall_auc.append(anthony)
    anthony_rmse = rmse(final_predictions_auc_correct, final_predictions_auc, average_over_labels=True)
    anthony_rmse_overall.append(anthony_rmse)
    anthony_kappa = cohen_kappa_multiclass(final_predictions_auc_correct, final_predictions_auc)
    anthony_overall_kappa.append(anthony_kappa)
    RMSE_0_traditional = math.sqrt(
        (sum((final_predictions_traditional['0.0_predicted_prob'] - final_predictions_traditional['0.0']) ** 2)) / len(
            final_predictions_traditional['0.0_predicted_prob']))
    RMSE_0_25_traditional = math.sqrt((sum(
        (final_predictions_traditional['0.25_predicted_prob'] - final_predictions_traditional['0.25']) ** 2)) / len(
        final_predictions_traditional['0.25_predicted_prob']))
    RMSE_0_5_traditional = math.sqrt(
        (sum((final_predictions_traditional['0.5_predicted_prob'] - final_predictions_traditional['0.5']) ** 2)) / len(
            final_predictions_traditional['0.5_predicted_prob']))
    RMSE_0_75_traditional = math.sqrt((sum(
        (final_predictions_traditional['0.75_predicted_prob'] - final_predictions_traditional['0.75']) ** 2)) / len(
        final_predictions_traditional['0.75_predicted_prob']))
    RMSE_1_traditional = math.sqrt(
        (sum((final_predictions_traditional['1.0_predicted_prob'] - final_predictions_traditional['1.0']) ** 2)) / len(
            final_predictions_traditional['1.0_predicted_prob']))
    RMSE_problem_grades_traditional = pd.DataFrame(pd.concat(
        [pd.Series(RMSE_0_traditional), pd.Series(RMSE_0_25_traditional), pd.Series(RMSE_0_5_traditional),
         pd.Series(RMSE_0_75_traditional), pd.Series(RMSE_1_traditional)], axis=1))
    Avg_RMSE_across_grades_traditional = pd.DataFrame(
        RMSE_problem_grades_traditional.mean(axis=1))  # , columns = ['Avg_RMSE_across_grades'])
    standard_error_traditional = pd.DataFrame(RMSE_problem_grades_traditional.sem(axis=1))
    RMSE_problem_grades_traditional = pd.DataFrame(
        pd.concat([RMSE_problem_grades_traditional, Avg_RMSE_across_grades_traditional], axis=1))
    RMSE_problem_grades_traditional = pd.DataFrame(
        pd.concat([RMSE_problem_grades_traditional, standard_error_traditional], axis=1))
    RMSE_problem_grades_traditional = pd.DataFrame(
        pd.concat([RMSE_problem_grades_traditional, pd.DataFrame(final_predictions_traditional.problem_id.unique())],
                  axis=1))
    RMSE_problem_grades_traditional = np.array(RMSE_problem_grades_traditional)
    RMSE_All_traditional.append(RMSE_problem_grades_traditional)
    RMSE_each_loop_traditional.append(RMSE_problem_grades_traditional)
    #                    pd.DataFrame(RMSE_each_loop)
    RMSE_from_loop_traditional = pd.DataFrame.from_records(RMSE_each_loop_traditional)
    RMSE_from_loop_traditional = pd.DataFrame(RMSE_from_loop_traditional[0].values.tolist(),
                                              columns=['0.0', '0.25', '0.5', '0.75', '1.0', 'average_RMSE',
                                                       'standard_error', 'problem_id_test'])
    RMSE_All_from_columns_traditional.append(np.array(RMSE_from_loop_traditional))
    RMSE_All_from_columns_data_frame_traditional = pd.DataFrame.from_records(RMSE_All_from_columns_traditional)
    RMSE_All_from_columns_data_frame_traditional = pd.DataFrame(
        RMSE_All_from_columns_data_frame_traditional[0].values.tolist(),
        columns=['0.0', '0.25', '0.5', '0.75', '1.0', 'average_RMSE', 'standard_error', 'problem_id_test'])
auc_result = np.mean(anthony_overall_auc)
print('Decision Tree AUC: ', auc_result)
rmse_dt_result = np.mean(anthony_rmse_overall)
print('Decision Tree RMSE: ',rmse_dt_result)
kappa_dt = np.mean(anthony_overall_kappa)
print('Decision Tree Kappa: ', kappa_dt)
print('time elapsed: {:.2f}s'.format(time.time() - start_time))




