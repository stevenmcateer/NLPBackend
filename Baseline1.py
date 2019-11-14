#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import math
import pandas as pd
import numpy as np
import ast
from numpy import zeros
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
import collections
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from datetime import datetime
# In[ ]:


df1=pd.read_csv('complete_data9thOct.csv')


# In[ ]:


print("Length:",len(df1.index))


# In[ ]:


print("Grades:",set(df1['grade']))


# In[ ]:


df1['parsed_cleaned_answers2']=[" ".join(ast.literal_eval(a)) for a in list(df1['parsed_cleaned_answers'])]
df1['parsed_cleaned_answers2'] = df1['parsed_cleaned_answers2'].str.replace(',','')
df1['parsed_cleaned_answers2'] = df1['parsed_cleaned_answers2'].str.replace(';','')
df1['parsed_cleaned_answers2'] = df1['parsed_cleaned_answers2'].str.replace('!','')
df1['parsed_cleaned_answers2'] =  df1['parsed_cleaned_answers2'].str.replace('','none')

# In[ ]:


df1=df1[['problem_id','folds','parsed_cleaned_answers','parsed_cleaned_answers2','grade','grade_0','grade_1','grade_2','grade_3','grade_4']]
#df1[0:5]


# In[ ]:


df1=df1[['problem_id','folds','parsed_cleaned_answers','parsed_cleaned_answers2','grade']]
df1 = pd.concat([df1, pd.get_dummies(df1['grade'],prefix='grade')], axis=1)
# df1[0:5]


# In[ ]:


#Tokenizer convert it to numbers
t = Tokenizer()
docs=df1['parsed_cleaned_answers2']
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
print("Vocab size",vocab_size)
# # integer encode the documents
#encoded_docs = t.texts_to_sequences(docs)

# #Converting to numpy array
#for i in range(0,len(encoded_docs )):
#     encoded_docs[i]=np.asarray(encoded_docs[i])
# encoded_docs=np.array(encoded_docs)


#Loading Glove
from numpy import asarray
embeddings_index = dict()
f = open('./glove.6B/glove.6B.100d.txt', 'rb')
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))



# # create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 100))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
       embedding_matrix[i] = embedding_vector


# In[ ]:
#lens=[len(x) for x in docs]
#e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=None, trainable=False)
            
e=Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=None, trainable=False,embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=False)
model = Sequential()
model.add(e)
model.add(LSTM(10,return_sequences=True))
model.add(LSTM(5, activation='softmax',return_sequences=False))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

model.save("InitialModel.h5")            





# In[ ]:


problem_rmse=[]
problem_acc=[]
def model_for_each_problem(pid, count, total_probs):
    problem_df=df1.loc[df1['problem_id']==pid]
    folds=list(set(problem_df['folds']))
    validate = False
    fold_rmse=[]
    fold_acc=[]
    if(len(folds)>=3):
        validate=True
        val_fold=folds[-1]
        print("More than 3 folds")



        folds.remove(val_fold)
    print("Folds:",folds)

    results=[]
    val_results=[] 
       
    for f in folds:
        print("Problem:", count, "/", total_probs)
        # current date and time
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        print("timestamp =", timestamp)
        actual=[]
        pred=[]
        train_df=problem_df[problem_df['folds']!=f]
        #print("Length of train:",len(train_df.index))      
        #Tokenizer convert it to numbers
        #t = Tokenizer()
        docs=train_df['parsed_cleaned_answers2']
        t.fit_on_texts(docs)
        #vocab_size = len(t.word_index) + 1
        # integer encode the documents
        train_encoded_docs = t.texts_to_sequences(docs)
            
        for i in range(0,len(train_encoded_docs )):
            train_encoded_docs[i]=np.asarray(train_encoded_docs[i])
        train_encoded_docs=np.array(train_encoded_docs)
            
            
        # create a weight matrix for words in training docs
        #embedding_matrix = zeros((vocab_size, 100))
        #for word, i in t.word_index.items():
        #    embedding_vector = embeddings_index.get(word)
        #    if embedding_vector is not None:
        #        embedding_matrix[i] = embedding_vector
            
        y_train=train_df[['grade_1','grade_2','grade_3','grade_4','grade_5']]
        y_train=np.asarray(y_train)
            
        # print("Train shape:",train_encoded_docs.shape)    
            
        #TEST DATA:
            
        test_df=problem_df[problem_df['folds']==f]
        print("Test Fold:",f)
        #print("Length of test df:",len(test_df.index))   
        print(test_df)
          
        docs=test_df['parsed_cleaned_answers2']
            
        #use same tokenizer
        test_encoded_docs=t.texts_to_sequences(docs)
            
        for i in range(0,len(test_encoded_docs )):
            test_encoded_docs[i]=np.asarray(test_encoded_docs[i])
        test_encoded_docs=np.array(test_encoded_docs)

        #print("Test shape:",test_encoded_docs.shape)
            
        y_test=test_df[['grade_1','grade_2','grade_3','grade_4','grade_5']]
        y_test=np.asarray(y_test)
            
            
            
            
        #VALIDATION DATA:
        if validate==True:
            
            val_df=problem_df[problem_df['folds']==val_fold]
            docs=val_df['parsed_cleaned_answers2']
            
            #use same tokenizer
            val_encoded_docs=t.texts_to_sequences(docs)
            
            for i in range(0,len(val_encoded_docs )):
                val_encoded_docs[i]=np.asarray(val_encoded_docs[i])
            val_x=np.array(val_encoded_docs)
           # print("Validate shape:",val_x.shape)
            
            y_val=val_df[['grade_1','grade_2','grade_3','grade_4','grade_5']]
            val_y=np.asarray(y_val)
            
            
        #define model
        model.load_weights("InitialModel.h5")            
        train_x,test_x = train_encoded_docs, test_encoded_docs
        train_y,test_y = y_train, y_test
        prev_a=1000000
        for index,val in enumerate(train_x):
            #print("train X:",train_x[index],"train y:",train_y[index])
            #print("Train index shape:",train_x[index].shape,"Train y shape:",train_y[index].shape)
            model.fit(train_x[index].reshape(1,len(val)),train_y[index].reshape(1,5),epochs = 10 ,batch_size=1)
            mean_a=[]
            if validate==True:
                
                for index,val in enumerate(val_x):
                    #print("VAL:",val)
                    print("VAL:",val_x.shape,val_y.shape)
                    a=model.evaluate(val_x[index].reshape(1,len(val)),val_y[index].reshape(1,5))
                    mean_a.append(a[0])
        #Checking if you should save these weights
                
                mean_a=np.mean(mean_a)
                print("Previous loss:",prev_a,"Current loss:",mean_a)
                if(mean_a<prev_a):
                    model.save_weights('./weights/val_weights_lstm.h5')
                    #print("Previous loss:",prev_a,"Current loss:",mean_a)
                      
                    print("saving")
                    prev_a=mean_a

                
        #If validate set exists, then use saved weights, else just use model. predict()         
        if(validate == True):
            model.load_weights('./weights/val_weights_lstm.h5')
        #model.fit(train_x,train_y,epochs = 100 ,batch_size=1)
        print("Test:",test_x)
        for index,val in enumerate(test_x):
            a=model.evaluate(test_x[index].reshape(1,len(val)),test_y[index].reshape(1,5))
            predictions=model.predict(test_x[index].reshape(1,len(val)))
            print("Pred:",predictions,"Argmax:",np.argmax(predictions,axis=1))
            actual.append(np.argmax(test_y[index].reshape(1,5)))
            pred.append(np.argmax(predictions,axis=1))
            
        actual=np.array(actual).flatten()
        pred=np.array(pred).flatten()
        print("Actual :",actual)
        print("Predictions:",pred)
        print("Fold:",f,"RMSE:",math.sqrt(mean_squared_error(actual,pred))) 
        print("Fold:",f,"Acc:",accuracy_score(actual,pred))
            
        results.append(a)
        fold_rmse.append(math.sqrt(mean_squared_error(actual,pred)))
        fold_acc.append(accuracy_score(actual,pred)) 
        # print("VALIDATION:")
            
        # for index,val in enumerate(val_x):
        #     a=model.evaluate(val_x[index].reshape(1,len(val)),val_y[index].reshape(1,5))
        #Checking if you should save these weights
        # prev_a=val_results[-1][0]
        # if(a[0]<prev_a):
        #     model.save_weights('./weights/val_weights_lstm.h5')
        #     print("saving")
        # val_results.append(a)
            
    results=np.asarray(results)
    fold_rmse=np.asarray(fold_rmse)
    fold_acc=np.asarray(fold_acc)
    print("problem id:",pid)
    print("Result:",results)
    print("RMSE:",fold_rmse)
    print("Mean RMSE:",np.mean(fold_rmse))
    print("Accuracy:",fold_acc)
    print("Mean Accuracy:",np.mean(fold_acc))
    problem_rmse.append(np.mean(fold_rmse))
    problem_acc.append(np.mean(fold_acc))
    print("RMSE so far:",np.mean(problem_rmse))
    print("Acc so far:",np.mean(problem_acc))
            
            
problem_ids=list(set(df1['problem_id']))
#model_for_each_problem(1228874)
count = 0
for p in problem_ids:
    count+=1
    print("Problem:", count, "/", len(problem_ids))
    model_for_each_problem(p, count, len(problem_ids))
    #model_for_each_problem(p)


print("Overall RMSE:",np.mean(problem_rmse))
print("Overall Acc:",np.mean(problem_acc)) 

