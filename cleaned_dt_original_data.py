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
        decision_tree_traditional = tree.DecisionTreeClassifier(max_depth=3)#, max_features='log2')#, class_weight='balanced')
        dt_fit_traditional = decision_tree_traditional.fit(term_freq_words_traditional, correct_traditional)
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
        filename = '.\Models\model-'+str(l)+'.sav'
        joblib.dump(dt_fit_traditional, filename)
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




################MAJORITY CALCULATIONS#####################
# majority_values = []
# rmse_majority = []
# ###calculate the majority
# for p in unique_ids_traditional:
#     problem = engage_ny.loc[engage_ny['problem_id'] == p]
#     majority = pd.DataFrame(problem.correct.mode())#.astype(int))
#     # np.array(majority)
#     majority_values.append(np.append(np.array(majority),[problem.problem_id.unique()]))
#     # majority2 = majority
#     # majority2['problem_id'] = problem.problem_id.unique()
#     #
#     #
#     # majority_values.append(majority2.values.tolist())
#     majority_error = pd.DataFrame(np.concatenate(problem['correct'].values-majority.values, axis = 0))
#     squared_error = (majority_error)**2
#     mean_squared_error = squared_error.mean()
#     root_mean_squared_error_majority = math.sqrt(mean_squared_error)
#     rmse_majority.append(root_mean_squared_error_majority)
#     # #for z in range(len(problem)):
#     #     data_majority = problem.loc[problem.index.values == problem.index.values[z]]
#     #     problem.correct.mode()
#     #     problem.correct
# avg_rmse_majority = pd.DataFrame(rmse_majority).mean()
#
# len(rmse_majority)
#
# pd.DataFrame(np.concatenate(majority_values, axis = 0))
# engage_ny.loc[engage_ny['problem_id'] == 1082509].correct.mode()


##BELOW IS EXCESS CODE, NOT NECESARY

#
#
#
# ####### SAVE PREDICTIONS FOR ENSEMBLE MODEL
# prediction_dataframe_save = pd.DataFrame(predictions_traditional).reset_index()
#         ##
#         ##                    #get the P(1) observations
# prediction_dataframe_save[5] = prediction_dataframe_save[5].str[0]
# prediction_dataframe_save[4] = prediction_dataframe_save[4].str[0]
# prediction_dataframe_save[4] = 1 - (prediction_dataframe_save[4].str[0])
# prediction_dataframe_save[3] = prediction_dataframe_save[3].str[0]
# prediction_dataframe_save[3] = 1 - (prediction_dataframe_save[3].str[0])
# prediction_dataframe_save[2] = prediction_dataframe_save[2].str[0]
# prediction_dataframe_save[2] = 1 - (prediction_dataframe_save[2].str[0])
# prediction_dataframe_save[1] = prediction_dataframe_save[1].str[0]
# prediction_dataframe_save[1] = 1 - (prediction_dataframe_save[1].str[0])
# prediction_dataframe_save[0] = prediction_dataframe_save[0].str[0]
# prediction_dataframe_save[0] = 1 - (prediction_dataframe_save[0].str[0])
# prediction_dataframe_save = prediction_dataframe_save.drop(columns=['index'])
#
# prediction_dataframe_save.columns = ['0.0_prediction_prob', '0.25_prediction_prob', '0.5_prediction_prob', '0.75_prediction_prob', '1.0_prediction_prob', 'problem_id']
# prediction_dataframe_save.to_csv('dt_results_2_7_19_original_data.csv', index = False)
#
#
#
# probs=pd.DataFrame(final_pred_prob_list).values.tolist()
# probs.to_csv('update_prob_dt.csv', index = False)
#
#
# #######################################################################################
# '''
# this is what will be used to plot the data
# '''
#
# RMSE_All_from_columns_data_frame_traditional_final = RMSE_All_from_columns_data_frame_traditional.drop(
#     ['0.0', '0.25', '0.5', '0.75', '1.0'], axis=1)
#
# '''
# '''
# RMSE_All_from_columns_data_frame_traditional_final.average_RMSE.mean()
#
#
#
#
# plot_tree_traditional = tree.export_graphviz(dt_fit_traditional, out_file=None)
# plot_the_tree = graphviz.Source(plot_tree_traditional)
# plot_the_tree.render()
# # RMSE_All_from_columns_data_frame_traditional = RMSE_All_from_columns_data_frame_traditional.drop(['sample_size'], axis = 1)
# #
# # RMSE_All_from_columns_data_frame_traditional.groupby()
# # float(RMSE_All_from_columns_data_frame_traditional['Avg_Avg_RMSE_from_loop'])
# # std_error_traditional= RMSE_All_from_columns_data_frame_traditional.problem_id_test.groupby('problem_id_test')['Avg_Avg_RMSE_from_loop'].mean().reset_index()
# #
# #
# #
# #
# #
# #
# #
# #
# #
# # Avg_Avg_RMSE_from_loop
#
# # RMSE_All_from_columns_data_frame_traditional.problem_id_test
# #
# # standard_error_plot_data_traditional = []
# # for c in RMSE_All_from_columns_data_frame_traditional.problem_id_test.unique():
# ##    RMSE_All_from_columns_data_frame.loc[RMSE_All_from_columns_data_frame['problem_id_test' == p]]
# #    problem_unique_Avg_Avg_RMSE_traditional = RMSE_All_from_columns_data_frame_traditional.loc[RMSE_All_from_columns_data_frame_traditional['problem_id_test']== c ]
# #    problem_id_number_traditional = problem_unique_Avg_Avg_RMSE_traditional.problem_id_test.unique()
# #    standard_error_traditional = problem_unique_Avg_Avg_RMSE_traditional.Avg_Avg_RMSE_from_loop.sem()
# #    problem_Avg_Avg_RMSE_mean_traditional = problem_unique_Avg_Avg_RMSE_traditional.Avg_Avg_RMSE_from_loop.mean()
# #    std_error_data_frame_traditional = pd.DataFrame(pd.concat([pd.Series(problem_Avg_Avg_RMSE_mean_traditional), pd.Series(standard_error_traditional),pd.Series(problem_id_number_traditional)], axis = 1))
# #    std_error_data_frame_traditional[2] = std_error_data_frame_traditional[2].astype('int')
# #    std_error_data_frame_traditional.astype('str').values.tolist()
# #    standard_error_plot_data_traditional.append(std_error_data_frame_traditional.values.tolist())
# #
#
#
# '''
# this eliminates the horizontal lines on the error bars
# '''
#
# matplotlib.rcParams.update({'errorbar.capsize': 0})
#
# # data_fram_pract_traditional = pd.DataFrame.from_records(np.array(standard_error_plot_data_traditional))
# # Final_STD_Error_Data_traditional = pd.DataFrame(data_fram_pract_traditional[0].values.tolist(), columns = ['Mean AVG AVG RMSE', 'STD Error', 'Problem_id'])
# # plt.errorbar(Final_STD_Error_Data_traditional.Problem_id.astype('str'), Final_STD_Error_Data_traditional['Mean AVG AVG RMSE'].sort_values(), yerr=Final_STD_Error_Data_traditional['STD Error'], fmt='o', color = 'k', ecolor = 'r', markersize=2)
# # plt.xticks([])
# # plt.xlabel('Problem')
# # plt.ylabel('RMSE Per Problem LOOCV')
# # plt.suptitle('Std Error of RMSE')
# #
#
# #
# # problem_1415683_traditional = Final_STD_Error_Data_traditional.loc[Final_STD_Error_Data_traditional['Problem_id'] == 1415683]
# # plt.plot(problem_1415683_traditional.Problem_id.astype('str'),problem_1415683_traditional['Mean AVG AVG RMSE'],marker='o', markersize=3, color="blue")
# #
# #
#
# '''
# maybe plot one more in the middle
# '''
# plot_extra = plt.figure()
# problem_1383722 = final_dataset_pandas.loc[final_dataset_pandas['problem_id_test'] == 1383722]
# problem_1383722_se = final_dataset_pandas_standard_error.loc[
#     final_dataset_pandas_standard_error['problem_id_test'] == 1383722]
# plt.errorbar(problem_1383722.sample_size.astype('str'), problem_1383722['average_RMSE'],
#              yerr=problem_1383722_se['standard_error'], fmt='o', color='k', ecolor='r', markersize=4)
# # plt.xticks(problem_max_error_diff.sample_size.astype('str'), problem_max_error_diff.sample_size.astype('str'))
# plt.xlabel('Sample_Size')
# plt.ylabel('RMSE for Problem')
# plt.suptitle('Std Error of RMSE Problem 1383722')
# plt.grid()
#
# plot_extra2 = plt.figure()
# problem_1460408 = final_dataset_pandas.loc[final_dataset_pandas['problem_id_test'] == 1460408]
# problem_1460408_se = final_dataset_pandas_standard_error.loc[
#     final_dataset_pandas_standard_error['problem_id_test'] == 1460408]
# plt.errorbar(problem_1460408.sample_size.astype('str'), problem_1460408['average_RMSE'],
#              yerr=problem_1460408_se['standard_error'], fmt='o', color='k', ecolor='r', markersize=4)
# # plt.xticks(problem_max_error_diff.sample_size.astype('str'), problem_max_error_diff.sample_size.astype('str'))
# plt.xlabel('Sample_Size')
# plt.ylabel('RMSE for Problem')
# plt.suptitle('Std Error of RMSE Problem 1460408')
# plt.grid()
#
# plot_extra3 = plt.figure()
# problem_1227509 = final_dataset_pandas.loc[final_dataset_pandas['problem_id_test'] == 1227509]
# problem_1227509_se = final_dataset_pandas_standard_error.loc[
#     final_dataset_pandas_standard_error['problem_id_test'] == 1227509]
# plt.errorbar(problem_1227509.sample_size.astype('str'), problem_1227509['average_RMSE'],
#              yerr=problem_1227509_se['standard_error'], fmt='o', color='k', ecolor='r', markersize=4)
# # plt.xticks(problem_max_error_diff.sample_size.astype('str'), problem_max_error_diff.sample_size.astype('str'))
# plt.xlabel('Sample_Size')
# plt.ylabel('RMSE for Problem')
# plt.suptitle('Std Error of RMSE Problem 1227509')
# plt.grid()
#
# '''
# this one is a good one below, clearly shows significant change from smallest sample to largest.
# CALCULATE P VALUE TO CONFIRM.
# '''
# #
# plot_extra4 = plt.figure()
# problem_1391043 = final_dataset_pandas.loc[final_dataset_pandas['problem_id_test'] == 1391043]
# problem_1391043_se = final_dataset_pandas_standard_error.loc[
#     final_dataset_pandas_standard_error['problem_id_test'] == 1391043]
# plt.errorbar(problem_1391043.sample_size.astype('str'), problem_1391043['average_RMSE'],
#              yerr=problem_1391043_se['standard_error'], fmt='o', color='k', ecolor='r', markersize=4)
# # plt.xticks(problem_max_error_diff.sample_size.astype('str'), problem_max_error_diff.sample_size.astype('str'))
# plt.xlabel('Sample_Size')
# plt.ylabel('RMSE for Problem')
# plt.suptitle('Std Error of RMSE Problem 1391043')
# plt.grid()
#
# plot_extra5 = plt.figure()
# problem_1391076 = final_dataset_pandas.loc[final_dataset_pandas['problem_id_test'] == 1391076]
# problem_1391076_se = final_dataset_pandas_standard_error.loc[
#     final_dataset_pandas_standard_error['problem_id_test'] == 1391076]
# plt.errorbar(problem_1391076.sample_size.astype('str'), problem_1391076['average_RMSE'],
#              yerr=problem_1391076_se['standard_error'], fmt='o', color='k', ecolor='r', markersize=4)
# # plt.xticks(problem_max_error_diff.sample_size.astype('str'), problem_max_error_diff.sample_size.astype('str'))
# plt.xlabel('Sample_Size')
# plt.ylabel('RMSE for Problem')
# plt.suptitle('Std Error of RMSE Problem 1391076')
# plt.grid()
#
# '''
# here we are going to plot the tradition LOOCV
# '''
#
# # data_fram_pract_traditional = pd.DataFrame.from_records(np.array(standard_error_plot_data_traditional))
# # Final_STD_Error_Data_traditional = pd.DataFrame(data_fram_pract_traditional[0].values.tolist(), columns = ['Mean AVG AVG RMSE', 'STD Error', 'Problem_id'])
# sorted_final_data = RMSE_All_from_columns_data_frame_traditional_final.sort_values(by=['average_RMSE'])
#
# plot_loocv = plt.figure()
# plt.errorbar(sorted_final_data.problem_id_test.astype('str'), sorted_final_data['average_RMSE'],
#              yerr=(2 * sorted_final_data['standard_error']), fmt='o', color='k', ecolor='r', markersize=2)
# plt.xticks([])
# plt.xlabel('Problem')
# plt.ylabel('RMSE Per Problem LOOCV')
# plt.suptitle('Confidence Intervals of RMSE with Trational LOOCV')
#
# '''
# here is where we put the star markers on the points
# '''
# # problem_1542052_traditional = sorted_final_data.loc[sorted_final_data['problem_id_test'] == 1542052]
# # problem_1082516_traditional = sorted_final_data.loc[sorted_final_data['problem_id_test'] == 1082516]
# # problem_1391885_traditional = sorted_final_data.loc[sorted_final_data['problem_id_test'] == 1391885]
# # problem_1391075_traditional = sorted_final_data.loc[sorted_final_data['problem_id_test'] == 1391075]
# # problem_1082509_traditional = sorted_final_data.loc[sorted_final_data['problem_id_test'] == 1082509]
# # problem_1212658_traditional = sorted_final_data.loc[sorted_final_data['problem_id_test'] == 1212658]
# # problem_1096508_traditional = sorted_final_data.loc[sorted_final_data['problem_id_test'] == 1096508]
# # problem_1249297_traditional = sorted_final_data.loc[sorted_final_data['problem_id_test'] == 1249297]
# # problem_1383722_traditional = sorted_final_data.loc[sorted_final_data['problem_id_test'] == 1383722]
# # problem_1460408_traditional = sorted_final_data.loc[sorted_final_data['problem_id_test'] == 1460408]
#
#
# # problem_1391042_traditional = sorted_final_data.loc[sorted_final_data['problem_id_test'] == 1391042]
# # problem_1391600_traditional = sorted_final_data.loc[sorted_final_data['problem_id_test'] == 1391600]
# problem_1460380_traditional = sorted_final_data.loc[sorted_final_data['problem_id_test'] == 1460380]  #
# problem_1415753_traditional = sorted_final_data.loc[sorted_final_data['problem_id_test'] == 1415753]  #
# problem_1487480_traditional = sorted_final_data.loc[sorted_final_data['problem_id_test'] == 1487480]  #
# problem_1415679_traditional = sorted_final_data.loc[sorted_final_data['problem_id_test'] == 1415679]  #
# # problem_1332620_traditional = sorted_final_data.loc[sorted_final_data['problem_id_test'] == 1332620]
# # problem_1423460_traditional = sorted_final_data.loc[sorted_final_data['problem_id_test'] == 1423460]
# problem_1391796_traditional = sorted_final_data.loc[sorted_final_data['problem_id_test'] == 1391796]  #
# problem_1423509_traditional = sorted_final_data.loc[sorted_final_data['problem_id_test'] == 1423509]  #
#
# #
# #
# # plt.plot(problem_1542052_traditional.problem_id_test.astype('str'),problem_1542052_traditional['average_RMSE'],marker='*', markersize=8, color="blue", label='Example A')
# # plt.plot(problem_1082516_traditional.problem_id_test.astype('str'),problem_1082516_traditional['average_RMSE'],marker='*', markersize=8, color="blue", label='Example A')
# # plt.plot(problem_1391885_traditional.problem_id_test.astype('str'),problem_1391885_traditional['average_RMSE'],marker='*', markersize=8, color="blue", label='Example A')
# # plt.plot(problem_1391075_traditional.problem_id_test.astype('str'),problem_1391075_traditional['average_RMSE'],marker='*', markersize=8, color="blue", label='Example A')
# # plt.plot(problem_1460380_traditional.problem_id_test.astype('str'),problem_1460380_traditional['average_RMSE'],marker='*', markersize=8, color="blue", label='Example A')
# # plt.plot(problem_1082509_traditional.problem_id_test.astype('str'),problem_1082509_traditional['average_RMSE'],marker='*', markersize=8, color="blue", label='Example A')
# #
# # plt.plot(problem_1212658_traditional.problem_id_test.astype('str'),problem_1212658_traditional['average_RMSE'],marker='*', markersize=8, color="blue", label='Example A')
# # plt.plot(problem_1096508_traditional.problem_id_test.astype('str'),problem_1096508_traditional['average_RMSE'],marker='*', markersize=8, color="blue", label='Example A')
# # plt.plot(problem_1249297_traditional.problem_id_test.astype('str'),problem_1249297_traditional['average_RMSE'],marker='*', markersize=8, color="blue", label='Example A')
# #
# # plt.plot(problem_1383722_traditional.problem_id_test.astype('str'),problem_1383722_traditional['average_RMSE'],marker='*', markersize=8, color="blue", label='Example A')
# # plt.plot(problem_1460408_traditional.problem_id_test.astype('str'),problem_1460408_traditional['average_RMSE'],marker='*', markersize=8, color="blue", label='Example A')
#
#
# # plt.plot(problem_1391042_traditional.problem_id_test.astype('str'),problem_1391042_traditional['average_RMSE'],marker='*', markersize=8, color="blue", label='Example A')
# # plt.plot(problem_1391600_traditional.problem_id_test.astype('str'),problem_1391600_traditional['average_RMSE'],marker='*', markersize=8, color="blue", label='Example A')
# plt.plot(problem_1415753_traditional.problem_id_test.astype('str'), problem_1415753_traditional['average_RMSE'],
#          marker='*', markersize=8, color="blue", label='Example A')
# plt.plot(problem_1487480_traditional.problem_id_test.astype('str'), problem_1487480_traditional['average_RMSE'],
#          marker='*', markersize=8, color="blue", label='Example A')
# plt.plot(problem_1415679_traditional.problem_id_test.astype('str'), problem_1415679_traditional['average_RMSE'],
#          marker='*', markersize=8, color="blue", label='Example A')
# plt.plot(problem_1460380_traditional.problem_id_test.astype('str'), problem_1460380_traditional['average_RMSE'],
#          marker='*', markersize=8, color="blue", label='Example A')
# # plt.plot(problem_1332620_traditional.problem_id_test.astype('str'),problem_1332620_traditional['average_RMSE'],marker='*', markersize=8, color="blue", label='Example A')
# # plt.plot(problem_1423460_traditional.problem_id_test.astype('str'),problem_1423460_traditional['average_RMSE'],marker='*', markersize=8, color="blue", label='Example A')
# plt.plot(problem_1391796_traditional.problem_id_test.astype('str'), problem_1391796_traditional['average_RMSE'],
#          marker='*', markersize=8, color="blue", label='Example A')
# plt.plot(problem_1423509_traditional.problem_id_test.astype('str'), problem_1423509_traditional['average_RMSE'],
#          marker='*', markersize=8, color="blue", label='Example A')
#
# sorted_final_data.loc[sorted_final_data['problem_id_test'] == 1096508]
#
# sorted_final_data.loc[sorted_final_data['Mean AVG AVG RMSE'] >= 0.3]
#
# # Plot the DT
#
# plot_tree = tree.export_graphviz(dt_fit, out_file=None)
# plot_the_tree = graphviz.Source(plot_tree)
#
# RMSE_from_loop
# RMSE_from_loop.to_csv('average_RMSE_at_each_grade.csv', index=False, header=True)  # make csv
# prediction_dataframe.to_csv('prediction_probability_example.csv', index=False, header=True)
#
# RMSE_All_from_columns_data_frame
#
# '''
#
# THIS WHOLE SECTION BELOW IS FOR PRINTING OUT THE ANSWERS FROM STUDENTS
#
#
# '''
#
# # 1331985
#
# print_prob_1487480 = engage_ny.loc[engage_ny['problem_id'] == 1487480]
# print_prob_1487480.to_csv('print_prob_1487480.csv', index=False, header=True)
#
# # 1096508
# print_prob_1096508 = engage_ny.loc[engage_ny['problem_id'] == 1096508].answer_text
# print_prob_1096508.to_csv('print_prob_1096508.csv', index=False, header=True)
#
# # 1415753
# print_prob_1415753 = engage_ny.loc[engage_ny['problem_id'] == 1415753]
# print_prob_1415753.to_csv('print_prob_1415753.csv', index=False, header=True)
#
# # 1460380
# print_prob_1460380 = engage_ny.loc[engage_ny['problem_id'] == 1460380]
# print_prob_1460380.to_csv('print_prob_1460380.csv', index=False, header=True)
#
# # 1415679
# print_prob_1415679 = engage_ny.loc[engage_ny['problem_id'] == 1415679]
# print_prob_1415679.to_csv('print_prob_1415679.csv', index=False, header=True)
#
# # 1391042
# print_prob_1391042 = engage_ny.loc[engage_ny['problem_id'] == 1391042].answer_text
# print_prob_1391042.to_csv('print_prob_1391042.csv', index=False, header=True)
#
# # 1391600
# print_prob_1391600 = engage_ny.loc[engage_ny['problem_id'] == 1391600].answer_text
# print_prob_1391600.to_csv('print_prob_1391600.csv', index=False, header=True)
#
# # 1423460
# print_prob_1423460 = engage_ny.loc[engage_ny['problem_id'] == 1423460].answer_text
# print_prob_1423460.to_csv('print_prob_1423460.csv', index=False, header=True)
#
# # 1542056
# print_prob_1542056 = engage_ny.loc[engage_ny['problem_id'] == 1542056].answer_text
# print_prob_1542056.to_csv('print_prob_1542056.csv', index=False, header=True)
#
# # 1391796
# print_prob_1391796 = engage_ny.loc[engage_ny['problem_id'] == 1391796].answer_text
# print_prob_1391796.to_csv('print_prob_1391796.csv', index=False, header=True)
#
# '''
# this gives you a range of RMSE values and their data
# '''
# sorted_final_data
# sorted_final_data[sorted_final_data[['average_RMSE']].apply(np.isclose, b=0.2, atol=0.02).any(1)]
#
# '''
# THIS IS FOR PRINTING THE DECISION TREES
# '''
#
# '''
# problem 1487480
# '''
#
# correct_orig = training_set.correct.astype('str')
#
# problem_1487480_data = engage_ny.loc[engage_ny['problem_id'] == 1487480]
# correct_dt_print = pd.merge(pd.DataFrame(problem_1487480_data.correct.astype('str')), dummy_predictors, left_index=True,
#                             right_index=True)
# correct_dt_print = correct_dt_print[[0.0, 0.25, 0.5, 0.75, 1.0]].astype('str')
# counting_tool_dt_print = CountVectorizer(tokenizer=TreebankWordTokenizer().tokenize,
#                                          analyzer='word')  ####TreebankWordTokenizer?
# counting_words_dt_print = counting_tool_dt_print.fit_transform(problem_1487480_data.answer_text)  # .split('\n')
# # counting_words_test = counting_tool.transform(text_x_value)
# # print(counting_words.shape)
# print(
#     counting_tool_dt_print.vocabulary_)  #### PRINTS OUT THE {'t': 33, 'h': 22, 'e': 19, ' ': 0, 'd': 18, 'i': 23, 'l': 25, 'a': 15, 'o': 28, 'n': 27, 's': 32, 'c': 17, 'g': 21, 'r': 31, 'u': 34, 'b': 16, 'p': 29, 'f': 20, 'm': 26, '.': 3, 'v': 35, '1': 6, '-': 2, '2': 7, 'q': 30, 'y': 37, 'k': 24, '5': 10, '0': 5, '÷': 38, '3': 8, '=': 14, '6': 11, '7': 12, '4': 9, '8': 13, 'w': 36, ',': 1, '/': 4}
# # counting_words
# ###calculate the term frequences #word/#total words
# term_freq_tool_dt_print = TfidfTransformer()
# term_freq_words_dt_print = term_freq_tool_dt_print.fit_transform(counting_words_dt_print)
# # term_freq_words_test = term_freq_tool.transform(counting_words_test)
# # Cross Validation standard LOOCV
# # nbc = MultinomialNB().fit(term_freq_words,correct)
# decision_tree_dt_print = tree.DecisionTreeClassifier(max_depth=3)  # tree.DecisionTreeClassifier(
# ###        min_samples_split=30, min_samples_leaf=10,
# ###        random_state=0)
# #                    dt_fit = decision_tree.fit(term_freq_words,correct_orig)
# dt_fit_dt_print = decision_tree_dt_print.fit(term_freq_words_dt_print, correct_dt_print)
#
# plot_tree_dt_print = tree.export_graphviz(dt_fit_dt_print, out_file=None,
#                                           feature_names=counting_tool_dt_print.get_feature_names())
# plot_the_tree_dt_print = graphviz.Source(plot_tree_dt_print)
#
# problem_1487480_data.problem_text
# problem_1487480_data.groupby('correct').count()
#
# problem_1487480_data.correct
#
# '''
# problem 1415679
# '''
#
# # correct_orig = training_set.correct  .astype('str')
#
#
# problem_1415679_data = engage_ny.loc[engage_ny['problem_id'] == 1415679]
# correct_dt_print_1415679 = pd.merge(pd.DataFrame(problem_1415679_data.correct.astype('str')), dummy_predictors,
#                                     left_index=True, right_index=True)
# correct_dt_print_1415679 = correct_dt_print_1415679[[0.0, 0.25, 0.5, 0.75, 1.0]].astype('str')
# counting_tool_dt_print_1415679 = CountVectorizer(tokenizer=TreebankWordTokenizer().tokenize,
#                                                  analyzer='word')  ####TreebankWordTokenizer?
# counting_words_dt_print_1415679 = counting_tool_dt_print_1415679.fit_transform(
#     problem_1415679_data.answer_text)  # .split('\n')
# # counting_words_test = counting_tool.transform(text_x_value)
# # print(counting_words.shape)
# print(
#     counting_tool_dt_print_1415679.vocabulary_)  #### PRINTS OUT THE {'t': 33, 'h': 22, 'e': 19, ' ': 0, 'd': 18, 'i': 23, 'l': 25, 'a': 15, 'o': 28, 'n': 27, 's': 32, 'c': 17, 'g': 21, 'r': 31, 'u': 34, 'b': 16, 'p': 29, 'f': 20, 'm': 26, '.': 3, 'v': 35, '1': 6, '-': 2, '2': 7, 'q': 30, 'y': 37, 'k': 24, '5': 10, '0': 5, '÷': 38, '3': 8, '=': 14, '6': 11, '7': 12, '4': 9, '8': 13, 'w': 36, ',': 1, '/': 4}
# # counting_words
# ###calculate the term frequences #word/#total words
# term_freq_tool_dt_print_1415679 = TfidfTransformer()
# term_freq_words_dt_print_1415679 = term_freq_tool_dt_print_1415679.fit_transform(counting_words_dt_print_1415679)
# # term_freq_words_test = term_freq_tool.transform(counting_words_test)
# # Cross Validation standard LOOCV
# # nbc = MultinomialNB().fit(term_freq_words,correct)
# decision_tree_dt_print_1415679 = tree.DecisionTreeClassifier(max_depth=2)  # tree.DecisionTreeClassifier(
# ###        min_samples_split=30, min_samples_leaf=10,
# ###        random_state=0)
# #                    dt_fit = decision_tree.fit(term_freq_words,correct_orig)
# dt_fit_dt_print_1415679 = decision_tree_dt_print_1415679.fit(term_freq_words_dt_print_1415679, correct_dt_print_1415679)
#
# plot_tree_dt_print_1415679 = tree.export_graphviz(dt_fit_dt_print_1415679, out_file=None,
#                                                   feature_names=counting_tool_dt_print_1415679.get_feature_names())
# plot_the_tree_dt_print_1415679 = graphviz.Source(plot_tree_dt_print_1415679)
#
# problem_1415679_data.groupby('correct').count()
#
# problem_1415679_data.correct
#
# '''
# problem 1415753
# '''
#
# # correct_orig = training_set.correct.astype('str')
#
#
# problem_1415753_data = engage_ny.loc[engage_ny['problem_id'] == 1415753]
# correct_dt_print_1415753 = pd.merge(pd.DataFrame(problem_1415753_data.correct.astype('str')), dummy_predictors,
#                                     left_index=True, right_index=True)
# correct_dt_print_1415753 = correct_dt_print_1415753[[0.0, 0.25, 0.5, 0.75, 1.0]].astype('str')
# counting_tool_dt_print_1415753 = CountVectorizer(tokenizer=TreebankWordTokenizer().tokenize,
#                                                  analyzer='word')  ####TreebankWordTokenizer?
# counting_words_dt_print_1415753 = counting_tool_dt_print_1415753.fit_transform(
#     problem_1415753_data.answer_text)  # .split('\n')
# # counting_words_test = counting_tool.transform(text_x_value)
# # print(counting_words.shape)
# print(
#     counting_tool_dt_print_1415753.vocabulary_)  #### PRINTS OUT THE {'t': 33, 'h': 22, 'e': 19, ' ': 0, 'd': 18, 'i': 23, 'l': 25, 'a': 15, 'o': 28, 'n': 27, 's': 32, 'c': 17, 'g': 21, 'r': 31, 'u': 34, 'b': 16, 'p': 29, 'f': 20, 'm': 26, '.': 3, 'v': 35, '1': 6, '-': 2, '2': 7, 'q': 30, 'y': 37, 'k': 24, '5': 10, '0': 5, '÷': 38, '3': 8, '=': 14, '6': 11, '7': 12, '4': 9, '8': 13, 'w': 36, ',': 1, '/': 4}
# # counting_words
# ###calculate the term frequences #word/#total words
# term_freq_tool_dt_print_1415753 = TfidfTransformer()
# term_freq_words_dt_print_1415753 = term_freq_tool_dt_print_1415753.fit_transform(counting_words_dt_print_1415753)
# # term_freq_words_test = term_freq_tool.transform(counting_words_test)
# # Cross Validation standard LOOCV
# # nbc = MultinomialNB().fit(term_freq_words,correct)
# decision_tree_dt_print_1415753 = tree.DecisionTreeClassifier(max_depth=3)  # tree.DecisionTreeClassifier(
# ###        min_samples_split=30, min_samples_leaf=10,
# ###        random_state=0)
# #                    dt_fit = decision_tree.fit(term_freq_words,correct_orig)
# dt_fit_dt_print_1415753 = decision_tree_dt_print_1415753.fit(term_freq_words_dt_print_1415753, correct_dt_print_1415753)
#
# plot_tree_dt_print_1415753 = tree.export_graphviz(dt_fit_dt_print_1415753, out_file=None,
#                                                   feature_names=counting_tool_dt_print_1415753.get_feature_names())
# plot_the_tree_dt_print_1415753 = graphviz.Source(plot_tree_dt_print_1415753)
#
# '''
# problem 1460380
# '''
#
# # correct_orig = training_set.correct.astype('str')
#
#
# problem_1460380_data = engage_ny.loc[engage_ny['problem_id'] == 1460380]
# correct_dt_print_1460380 = pd.merge(pd.DataFrame(problem_1460380_data.correct.astype('str')), dummy_predictors,
#                                     left_index=True, right_index=True)
# correct_dt_print_1460380 = correct_dt_print_1460380[[0.0, 0.25, 0.5, 0.75, 1.0]].astype('str')
# counting_tool_dt_print_1460380 = CountVectorizer(tokenizer=TreebankWordTokenizer().tokenize,
#                                                  analyzer='word')  ####TreebankWordTokenizer?
# counting_words_dt_print_1460380 = counting_tool_dt_print_1460380.fit_transform(
#     problem_1460380_data.answer_text)  # .split('\n')
# # counting_words_test = counting_tool.transform(text_x_value)
# # print(counting_words.shape)
# print(
#     counting_tool_dt_print_1460380.vocabulary_)  #### PRINTS OUT THE {'t': 33, 'h': 22, 'e': 19, ' ': 0, 'd': 18, 'i': 23, 'l': 25, 'a': 15, 'o': 28, 'n': 27, 's': 32, 'c': 17, 'g': 21, 'r': 31, 'u': 34, 'b': 16, 'p': 29, 'f': 20, 'm': 26, '.': 3, 'v': 35, '1': 6, '-': 2, '2': 7, 'q': 30, 'y': 37, 'k': 24, '5': 10, '0': 5, '÷': 38, '3': 8, '=': 14, '6': 11, '7': 12, '4': 9, '8': 13, 'w': 36, ',': 1, '/': 4}
# # counting_words
# ###calculate the term frequences #word/#total words
# term_freq_tool_dt_print_1460380 = TfidfTransformer()
# term_freq_words_dt_print_1460380 = term_freq_tool_dt_print_1460380.fit_transform(counting_words_dt_print_1460380)
# # term_freq_words_test = term_freq_tool.transform(counting_words_test)
# # Cross Validation standard LOOCV
# # nbc = MultinomialNB().fit(term_freq_words,correct)
# decision_tree_dt_print_1460380 = tree.DecisionTreeClassifier(max_depth=3)  # tree.DecisionTreeClassifier(
# ###        min_samples_split=30, min_samples_leaf=10,
# ###        random_state=0)
# #                    dt_fit = decision_tree.fit(term_freq_words,correct_orig)
# dt_fit_dt_print_1460380 = decision_tree_dt_print_1460380.fit(term_freq_words_dt_print_1460380, correct_dt_print_1460380)
#
# plot_tree_dt_print_1460380 = tree.export_graphviz(dt_fit_dt_print_1460380, out_file=None,
#                                                   feature_names=counting_tool_dt_print_1460380.get_feature_names())
# plot_the_tree_dt_print_1460380 = graphviz.Source(plot_tree_dt_print_1460380)
#
# problem_1460380_data.groupby('correct').count()
#
# problem_1460380_data.problem_text
#
# '''
# problem 1391600
# '''
#
# # correct_orig = training_set.correct.astype('str')
#
#
# problem_1391600_data = engage_ny.loc[engage_ny['problem_id'] == 1391600]
# correct_dt_print_1391600 = pd.merge(pd.DataFrame(problem_1391600_data.correct.astype('str')), dummy_predictors,
#                                     left_index=True, right_index=True)
# correct_dt_print_1391600 = correct_dt_print_1391600[[0.0, 0.25, 0.5, 0.75, 1.0]].astype('str')
# counting_tool_dt_print_1391600 = CountVectorizer(tokenizer=TreebankWordTokenizer().tokenize,
#                                                  analyzer='word')  ####TreebankWordTokenizer?
# counting_words_dt_print_1391600 = counting_tool_dt_print_1391600.fit_transform(
#     problem_1391600_data.answer_text)  # .split('\n')
# # counting_words_test = counting_tool.transform(text_x_value)
# # print(counting_words.shape)
# print(
#     counting_tool_dt_print_1391600.vocabulary_)  #### PRINTS OUT THE {'t': 33, 'h': 22, 'e': 19, ' ': 0, 'd': 18, 'i': 23, 'l': 25, 'a': 15, 'o': 28, 'n': 27, 's': 32, 'c': 17, 'g': 21, 'r': 31, 'u': 34, 'b': 16, 'p': 29, 'f': 20, 'm': 26, '.': 3, 'v': 35, '1': 6, '-': 2, '2': 7, 'q': 30, 'y': 37, 'k': 24, '5': 10, '0': 5, '÷': 38, '3': 8, '=': 14, '6': 11, '7': 12, '4': 9, '8': 13, 'w': 36, ',': 1, '/': 4}
# # counting_words
# ###calculate the term frequences #word/#total words
# term_freq_tool_dt_print_1391600 = TfidfTransformer()
# term_freq_words_dt_print_1391600 = term_freq_tool_dt_print_1391600.fit_transform(counting_words_dt_print_1391600)
# # term_freq_words_test = term_freq_tool.transform(counting_words_test)
# # Cross Validation standard LOOCV
# # nbc = MultinomialNB().fit(term_freq_words,correct)
# decision_tree_dt_print_1391600 = tree.DecisionTreeClassifier(max_depth=2)  # tree.DecisionTreeClassifier(
# ###        min_samples_split=30, min_samples_leaf=10,
# ###        random_state=0)
# #                    dt_fit = decision_tree.fit(term_freq_words,correct_orig)
# dt_fit_dt_print_1391600 = decision_tree_dt_print_1391600.fit(term_freq_words_dt_print_1391600, correct_dt_print_1391600)
#
# plot_tree_dt_print_1391600 = tree.export_graphviz(dt_fit_dt_print_1391600, out_file=None)
# plot_the_tree_dt_print_1391600 = graphviz.Source(plot_tree_dt_print_1391600)
#
# '''
# problem 1391042
# '''
#
# # correct_orig = training_set.correct.astype('str')
#
#
# problem_1391042_data = engage_ny.loc[engage_ny['problem_id'] == 1391042]
# correct_dt_print_1391042 = pd.merge(pd.DataFrame(problem_1391042_data.correct.astype('str')), dummy_predictors,
#                                     left_index=True, right_index=True)
# correct_dt_print_1391042 = correct_dt_print_1391042[[0.0, 0.25, 0.5, 0.75, 1.0]].astype('str')
# counting_tool_dt_print_1391042 = CountVectorizer(tokenizer=TreebankWordTokenizer().tokenize,
#                                                  analyzer='word')  ####TreebankWordTokenizer?
# counting_words_dt_print_1391042 = counting_tool_dt_print_1391042.fit_transform(
#     problem_1391042_data.answer_text)  # .split('\n')
# # counting_words_test = counting_tool.transform(text_x_value)
# # print(counting_words.shape)
# print(
#     counting_tool_dt_print_1391042.vocabulary_)  #### PRINTS OUT THE {'t': 33, 'h': 22, 'e': 19, ' ': 0, 'd': 18, 'i': 23, 'l': 25, 'a': 15, 'o': 28, 'n': 27, 's': 32, 'c': 17, 'g': 21, 'r': 31, 'u': 34, 'b': 16, 'p': 29, 'f': 20, 'm': 26, '.': 3, 'v': 35, '1': 6, '-': 2, '2': 7, 'q': 30, 'y': 37, 'k': 24, '5': 10, '0': 5, '÷': 38, '3': 8, '=': 14, '6': 11, '7': 12, '4': 9, '8': 13, 'w': 36, ',': 1, '/': 4}
# # counting_words
# ###calculate the term frequences #word/#total words
# term_freq_tool_dt_print_1391042 = TfidfTransformer()
# term_freq_words_dt_print_1391042 = term_freq_tool_dt_print_1391042.fit_transform(counting_words_dt_print_1391042)
# # term_freq_words_test = term_freq_tool.transform(counting_words_test)
# # Cross Validation standard LOOCV
# # nbc = MultinomialNB().fit(term_freq_words,correct)
# decision_tree_dt_print_1391042 = tree.DecisionTreeClassifier(max_depth=2)  # tree.DecisionTreeClassifier(
# ###        min_samples_split=30, min_samples_leaf=10,
# ###        random_state=0)
# #                    dt_fit = decision_tree.fit(term_freq_words,correct_orig)
# dt_fit_dt_print_1391042 = decision_tree_dt_print_1391042.fit(term_freq_words_dt_print_1391042, correct_dt_print_1391042)
#
# plot_tree_dt_print_1391042 = tree.export_graphviz(dt_fit_dt_print_1391042, out_file=None)
# plot_the_tree_dt_print_1391042 = graphviz.Source(plot_tree_dt_print_1391042)
#
# '''
# problem 1423460
# '''
#
# # correct_orig = training_set.correct.astype('str')
#
#
# problem_1423460_data = engage_ny.loc[engage_ny['problem_id'] == 1423460]
# correct_dt_print_1423460 = pd.merge(pd.DataFrame(problem_1423460_data.correct.astype('str')), dummy_predictors,
#                                     left_index=True, right_index=True)
# correct_dt_print_1423460 = correct_dt_print_1423460[[0.0, 0.25, 0.5, 0.75, 1.0]].astype('str')
# counting_tool_dt_print_1423460 = CountVectorizer(tokenizer=TreebankWordTokenizer().tokenize,
#                                                  analyzer='word')  ####TreebankWordTokenizer?
# counting_words_dt_print_1423460 = counting_tool_dt_print_1423460.fit_transform(
#     problem_1423460_data.answer_text)  # .split('\n')
# # counting_words_test = counting_tool.transform(text_x_value)
# # print(counting_words.shape)
# print(
#     counting_tool_dt_print_1423460.vocabulary_)  #### PRINTS OUT THE {'t': 33, 'h': 22, 'e': 19, ' ': 0, 'd': 18, 'i': 23, 'l': 25, 'a': 15, 'o': 28, 'n': 27, 's': 32, 'c': 17, 'g': 21, 'r': 31, 'u': 34, 'b': 16, 'p': 29, 'f': 20, 'm': 26, '.': 3, 'v': 35, '1': 6, '-': 2, '2': 7, 'q': 30, 'y': 37, 'k': 24, '5': 10, '0': 5, '÷': 38, '3': 8, '=': 14, '6': 11, '7': 12, '4': 9, '8': 13, 'w': 36, ',': 1, '/': 4}
# # counting_words
# ###calculate the term frequences #word/#total words
# term_freq_tool_dt_print_1423460 = TfidfTransformer()
# term_freq_words_dt_print_1423460 = term_freq_tool_dt_print_1423460.fit_transform(counting_words_dt_print_1423460)
# # term_freq_words_test = term_freq_tool.transform(counting_words_test)
# # Cross Validation standard LOOCV
# # nbc = MultinomialNB().fit(term_freq_words,correct)
# decision_tree_dt_print_1423460 = tree.DecisionTreeClassifier(max_depth=2)  # tree.DecisionTreeClassifier(
# ###        min_samples_split=30, min_samples_leaf=10,
# ###        random_state=0)
# #                    dt_fit = decision_tree.fit(term_freq_words,correct_orig)
# dt_fit_dt_print_1423460 = decision_tree_dt_print_1423460.fit(term_freq_words_dt_print_1423460, correct_dt_print_1423460)
#
# plot_tree_dt_print_1423460 = tree.export_graphviz(dt_fit_dt_print_1423460, out_file=None)
# plot_the_tree_dt_print_1423460 = graphviz.Source(plot_tree_dt_print_1423460)
#
# '''
# problem 1391796
# '''
#
# # correct_orig = training_set.correct.astype('str')
#
# # problem_1391796_data.shape
#
# problem_1391796_data = engage_ny.loc[engage_ny['problem_id'] == 1391796]
# correct_dt_print_1391796 = pd.merge(pd.DataFrame(problem_1391796_data.correct.astype('str')), dummy_predictors,
#                                     left_index=True, right_index=True)
# correct_dt_print_1391796 = correct_dt_print_1391796[[0.0, 0.25, 0.5, 0.75, 1.0]].astype('str')
# counting_tool_dt_print_1391796 = CountVectorizer(tokenizer=TreebankWordTokenizer().tokenize,
#                                                  analyzer='word')  ####TreebankWordTokenizer?
# counting_words_dt_print_1391796 = counting_tool_dt_print_1391796.fit_transform(
#     problem_1391796_data.answer_text)  # .split('\n')
# # counting_words_test = counting_tool.transform(text_x_value)
# # print(counting_words.shape)
# print(
#     counting_tool_dt_print_1391796.vocabulary_)  #### PRINTS OUT THE {'t': 33, 'h': 22, 'e': 19, ' ': 0, 'd': 18, 'i': 23, 'l': 25, 'a': 15, 'o': 28, 'n': 27, 's': 32, 'c': 17, 'g': 21, 'r': 31, 'u': 34, 'b': 16, 'p': 29, 'f': 20, 'm': 26, '.': 3, 'v': 35, '1': 6, '-': 2, '2': 7, 'q': 30, 'y': 37, 'k': 24, '5': 10, '0': 5, '÷': 38, '3': 8, '=': 14, '6': 11, '7': 12, '4': 9, '8': 13, 'w': 36, ',': 1, '/': 4}
# # counting_words
# ###calculate the term frequences #word/#total words
# term_freq_tool_dt_print_1391796 = TfidfTransformer()
# term_freq_words_dt_print_1391796 = term_freq_tool_dt_print_1391796.fit_transform(counting_words_dt_print_1391796)
# # term_freq_words_test = term_freq_tool.transform(counting_words_test)
# # Cross Validation standard LOOCV
# # nbc = MultinomialNB().fit(term_freq_words,correct)
# decision_tree_dt_print_1391796 = tree.DecisionTreeClassifier(max_depth=3)  # tree.DecisionTreeClassifier(
# ###        min_samples_split=30, min_samples_leaf=10,
# ###        random_state=0)
# #                    dt_fit = decision_tree.fit(term_freq_words,correct_orig)
# dt_fit_dt_print_1391796 = decision_tree_dt_print_1391796.fit(term_freq_words_dt_print_1391796, correct_dt_print_1391796)
#
# plot_tree_dt_print_1391796 = tree.export_graphviz(dt_fit_dt_print_1391796, out_file=None,
#                                                   feature_names=counting_tool_dt_print_1391796.get_feature_names())
# plot_the_tree_dt_print_1391796 = graphviz.Source(plot_tree_dt_print_1391796)
#
# # counting_tool_dt_print_1391796.get_feature_names() this gives you the variales
#
# problem_1391796_data.groupby('correct').count()
#
# #        dt_fit_traditional = decision_tree_traditional.fit(term_freq_words_traditional,correct_traditional)
# #        tree_predict_probability_traditional = dt_fit_traditional.predict_proba(term_freq_words_test_traditional)###predict the probabilities
#
#
# problem_1391796_data.problem_text
#
# '''
# test
# '''
# problem_1391796_datatest = engage_ny.loc[engage_ny['problem_id'] == 1391796].sample(1)
# # correct_dt_print_1391796test=pd.merge(pd.DataFrame(problem_1391796_datatest.correct.astype('str')), dummy_predictors, left_index=True, right_index=True)
# # correct_dt_print_1391796test= correct_dt_print_1391796test[[0.0, 0.25, 0.5, 0.75, 1.0]].astype('str')
# # counting_tool_dt_print_1391796test= CountVectorizer(tokenizer=TreebankWordTokenizer().tokenize, analyzer='word')####TreebankWordTokenizer?
# counting_words_dt_print_1391796test = counting_tool_dt_print_1391796.transform(
#     problem_1391796_datatest.answer_text)  # .split('\n')
# term_freq_tool_dt_print_1391796test = TfidfTransformer(use_idf=True)
# term_freq_words_dt_print_1391796test = term_freq_tool_dt_print_1391796test.fit_transform(
#     counting_words_dt_print_1391796test)
#
# dt_fit_dt_print_1391796
# tree_predict_probability_traditional = dt_fit_dt_print_1391796.predict_proba(
#     term_freq_words_dt_print_1391796test)  ###predict the probabilities
#
# term_freq_tool_dt_print_1391796test.idf_
#
# # tree_predict_traditional = dt_fit_traditional.predict(term_freq_words_test_traditional)###predict the probabilities
#
# # weighted_values = term_freq_words_dt_print_1391796test.mean(axis = 0).ravel().tolist()
# #
# # weighted_values_pd = pd.DataFrame({'term': counting_words_dt_print_1391796test.get_feature_names(), 'weight': weights})
# #
# #
# #
# #
# # counting_tool = CountVectorizer(tokenizer=TreebankWordTokenizer().tokenize, analyzer='word')####TreebankWordTokenizer?
# #                    counting_words=counting_tool.fit_transform(answers)#.split('\n')
# #                    counting_words_test = counting_tool.transform(text_x_value)
# #                    #print(counting_words.shape)
# #                #print(counting_tool.vocabulary_)#### PRINTS OUT THE {'t': 33, 'h': 22, 'e': 19, ' ': 0, 'd': 18, 'i': 23, 'l': 25, 'a': 15, 'o': 28, 'n': 27, 's': 32, 'c': 17, 'g': 21, 'r': 31, 'u': 34, 'b': 16, 'p': 29, 'f': 20, 'm': 26, '.': 3, 'v': 35, '1': 6, '-': 2, '2': 7, 'q': 30, 'y': 37, 'k': 24, '5': 10, '0': 5, '÷': 38, '3': 8, '=': 14, '6': 11, '7': 12, '4': 9, '8': 13, 'w': 36, ',': 1, '/': 4}
# #                    counting_words
# #                ###calculate the term frequences #word/#total words
# #                    term_freq_tool = TfidfTransformer()
# #                    term_freq_words = term_freq_tool.fit_transform(counting_words)
# #                    term_freq_words_test = term_freq_tool.transform(counting_words_test)
# #
#
#
# engage_ny.groupby('user_id').count()  # 816 unique users
#
# engage_ny.loc[engage_ny['problem_id'] == 1391885].correct
#
# '''
# predict with just majority
# '''
#
# rmse_majority_each_problem = []
# errors_data = []
# majority_class_data = []
# problem_id_data = []
# for v in engage_ny.problem_id.unique():
#     selected_se_problem = engage_ny.loc[engage_ny['problem_id'] == v]
#     #    correct_val = selected_se_problem.correct
#     majority_class = selected_se_problem.correct.value_counts().idxmax()
#     #    RMSE_majority= (((selected_se_problem.correct - majority_class)**2).mean())**(1/2)
#     #    rmse_majority_each_problem.append(RMSE_majority)
#     majority_class_data.append(majority_class)
#     problem_id_data.append(selected_se_problem.problem_id.unique())
#     for vr in range(len(selected_se_problem)):
#         test_set_traditional = engage_ny.loc[engage_ny.index.values == engage_ny.index.values[vr]]
#         test_y_value_traditional = test_set_traditional.correct
#         error = (majority_class - test_y_value_traditional) ** 2
#         errorpd = pd.DataFrame(
#             pd.concat([pd.Series(np.array(error)), pd.Series(selected_se_problem.problem_id.unique())], axis=1))
#         #        errorpd = pd.concat([pd.Series(error), pd.Series(selected_se_problem.problem_id.unique())], axis=1)
#         #        np.array(errorpd)
#         errors_data.append(np.array(errorpd))
# #        errors_data.extend([np.array(selected_se_problem.problem_id.unique())])
# #        dimensions_traditional = engage_ny_uniq_id_traditional.drop(test_set_traditional.index).shape[0]
# #        training_set_traditional = majority_class
# # RMSE_majority= (((test_y_value_traditional- (majority_class))**2).mean())**(1/2)
# # rmse_majority_each_problem.append(RMSE_majority)
#
# errors_data
#
# # pd.DataFrame(errors_data)..split(' ',1).tolist()
#
# error_data_pd = pd.DataFrame.from_records(errors_data)
# error_data_pd = pd.DataFrame(error_data_pd[0].values.tolist(), columns=['sq_error', 'problem_id'])
# mean_sq_error = error_data_pd.groupby('problem_id').mean().reset_index()
# ((mean_sq_error.sq_error) ** (1 / 2)).mean()
