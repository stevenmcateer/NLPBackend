from sklearn.externals import joblib

def calculateGrade(problemId, response):
	filename = '.\Terms\model-'+str(problemId)+'.sav'
	term_freq_tool_traditional = joblib.load(filename)
	filename = '.\Models\model-'+str(problemId)+'.sav'
	dt_fit_traditional = joblib.load(filename)
	filename = '.\Counting\model-'+str(problemId)+'.sav'
	counting_tool_traditional = joblib.load(filename)
	counting_words_test_traditional =counting_tool_traditional.transform([str(response)])
	term_freq_words_test_traditional = term_freq_tool_traditional.transform(counting_words_test_traditional)
	tree_predict_traditional = dt_fit_traditional.predict(
        term_freq_words_test_traditional)
	tree_predict_probability_traditional = dt_fit_traditional.predict_proba(
        term_freq_words_test_traditional)
	#print('Predict: ', tree_predict_traditional)
	#print('Probability: ',tree_predict_probability_traditional)
	#grade = 0
	if tree_predict_traditional[0][0]==1:
		grade=0
	if tree_predict_traditional[0][1]==1:
		grade=1
	if tree_predict_traditional[0][2]==1:
		grade=2
	if tree_predict_traditional[0][3]==1:
		grade=3
	if tree_predict_traditional[0][4]==1:
		grade=4
	#print('Grade: ',grade)
	return grade



problemId= 1276708
response = "There are 38 girls"
calculateGrade(problemId, response)