import numpy as np 
import re
import pandas as pd
from nltk.corpus import stopwords
import sklearn
from collections import Counter
from nltk.tokenize import word_tokenize
import collections
from sklearn.model_selection import train_test_split

data=pd.read_csv('Tweets.csv')
data= data.copy()[['airline_sentiment', 'text']]
# remove the punction and stopwords
def review_to_words( review ):
    review_text = review
    no_hasthtags = re.sub("#\w+", " ", review_text)
    no_url = re.sub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", " ", no_hasthtags )
    no_tag = re.sub("@\w+", " ", no_url)
    no_punctions = re.sub("[^a-zA-Z]", " ", no_tag) 
    wordslower= no_punctions.lower()
    words = word_tokenize(wordslower)  
    stopswd = set(stopwords.words("english"))                  
    meaningful_wd = [w for w in words if not w in stopswd]
    return(meaningful_wd)

clean_text = []
for tweet in data['text']:
    clean= review_to_words(tweet)
    clean_text.append(clean)

data['text'] = clean_text
# 90% of data is data_train for traing,10% of data is data_test for testing

def cross_validation(cv=10, avg=True):
	k = [int((len(data['text']))/cv*j) for j in range(cv+1)]
#k=[0, 1464, 2928, 4392, 5856, 7320, 8784, 10248, 11712, 13176, 14640]
#We devide 14640 Tweets into 10 piles. 
        o=0
        for t in range (cv):# 10-fold CV
                print t
                X_test, y_test= data['text'][k[t]:k[t+1]], data['airline_sentiment'][k[t]:k[t+1]]
#we choose the 10% of the data as the test data, and the other 90% of the data as the traing data
		X_train, y_train = pd.concat([data['text'][:k[t]],data['text'][k[t+1]:]]), pd.concat([data['airline_sentiment'][:k[t]],data['airline_sentiment'][k[t+1]:]])
# d is the number of words in positive text,I remember d= 16650
# e is the number of words in negative text,I remember e= 80297
# f is is the number of words in neutral text,I remember f= 21642	
                u=0
		a=0
		b=0
		c=0
		e=0
		d=0
		f=0
		dataAll=X_train
		copos1=collections.Counter()
		coneg2=collections.Counter()
		coneu3=collections.Counter() 
		dataPos=X_train.copy()[y_train== 'positive']
		dataNeg=X_train.copy()[y_train== 'negative']
		dataNeu=X_train.copy()[y_train== 'neutral']
		pos_words =(dataPos.tolist())
		neg_words =(dataNeg.tolist())
		neu_words =(dataNeu.tolist())
		all_words =(X_train.tolist())
		P_pos = float(len(dataPos))/len(dataAll)
		P_neg = float(len(dataNeg))/len(dataAll)
		P_neu = float(len(dataNeu))/len(dataAll)
# I use copos,coneg,coneu to divide the training data into 3 emotions
		for i in range(0, len(pos_words)):
		        copos1.update(pos_words[i])

		for i in range(0, len(neg_words)):
		        coneg2.update(neg_words[i])

		for i in range(0, len(neu_words)): 
		        coneu3.update(neu_words[i])

		voctot = collections.Counter()
		voctot.update(copos1)
		voctot.update(coneg2)
		voctot.update(coneu3)
		nvoctot=len(voctot)
#calcul the number d,e,f
                d=sum(len(x) for x in pos_words)
                e=sum(len(x) for x in neg_words)
                f=sum(len(x) for x in neu_words)
		print("nmots pos neg neu",d,e,f)
#then I predict the result of Testing data
# the result of predicting including positive,negative and neutral
		class_choice = ['positive', 'negative', 'neutral']
		classification = []
		test_words =(X_test.tolist())
# i= the i-th person'sopinion
		for i in range(0, len(test_words)):
#each word in i-th person's opinion
      		       for w in test_words[i]:
                                  a=a+np.log((float(copos1[w]+1))/float(d+nvoctot))
		                  b=b+np.log((float(coneg2[w]+1))/float(e+nvoctot))
		                  c=c+np.log((float(coneu3[w]+1))/float(f+nvoctot))
# il suffit de calculer ln P(S|D) = sum_w ln(P(w|S)) + ln(P(S)) - constante
# et on se moque de la constante
		       a=a+np.log(P_pos)
                       b=b+np.log(P_neg)
                       c=c+np.log(P_neu)
#I choose the best results from the training data to predict the testing data
                       probability = (a, b, c)
                       classification.append(class_choice[np.argmax(probability)])
                a=0
                b=0
                c=0
#I calcul the accuracy
		compare = []
		for i in range(0,len(classification)):
		            if classification[i] == y_test.tolist()[i]:
		                value ='correct'
		                compare.append(value)
		            else:
		                value ='incorrect'
		                compare.append(value)

		r = Counter(compare)
		accuracy = float(r['correct'])/float(r['correct']+r['incorrect'])
                print ("accuracy:",accuracy)
                o=o+accuracy

                for z in range(0,len(classification)):
                    if classification[z] == y_test.tolist()[z]:
                                u=u+1
                print ("number of correct predict Tweet in traing data:",u)      
        return o/10

avg_score = [cross_validation(avg=True,cv=10)]
print "average accuracy is"
print(avg_score)

