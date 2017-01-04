import numpy as np 
import re
import pandas as pd
from nltk.corpus import stopwords
from collections import Counter
from nltk.tokenize import word_tokenize
import collections
 

data=pd.read_csv("Tweets.csv")
  
def review_to_words( review ):
    review_text = review
    no_punctions = re.sub("[^a-zA-Z]", " ", review_text) 
    wordslower= no_punctions.lower()
    words = word_tokenize(wordslower)  
    stopswd = set(stopwords.words("english"))                  
    meaningful_wd = [w for w in words if not w in stopswd]
    return(meaningful_wd)

Posdata_list = []
Negdata_list = []
Neudata_list = []

ReviewsPos=[]
copos=collections.Counter()         
dataPos=data[data['airline_sentiment']=='positive']
for i in range(0, len(dataPos)):
    ReviewsPos.append(review_to_words(dataPos['text'].tolist()[i]))
    copos.update(ReviewsPos[i])       

for w in copos.keys(): 
          Posdata_list.append((w,str(copos[w]/float(len(dataPos))))) 
Pos=sorted(Posdata_list, key=lambda x:x[1],reverse=True)
print "******************************"
print "Positive 20"
print "******************************"
for i in range(0,20):
     print Pos[i][0]


ReviewsNeg=[]
coneg=collections.Counter()        
dataNeg=data[data['airline_sentiment']=='negative']
for i in range(0, len(dataNeg)):
    ReviewsNeg.append(review_to_words(dataNeg['text'].tolist()[i]))
    coneg.update(ReviewsNeg[i])      

for w in coneg.keys():
      Negdata_list.append((w,str(coneg[w]/float(len(dataNeg))))) 
Neg=sorted(Negdata_list, key=lambda x:x[1],reverse=True)
print "******************************"
print "Negative 20"
print "******************************"
for i in range(0,20):
     print Neg[i][0]




ReviewsNeutre=[]
coneu=collections.Counter()           
dataNeutre=data[data['airline_sentiment']=='neutral']
for i in range(0, len(dataNeutre)):
    ReviewsNeutre.append(review_to_words(dataNeutre['text'].tolist()[i]))
    coneu.update(ReviewsNeutre[i])       

for w in coneu.keys(): 
      Neudata_list.append((w,str(coneu[w]/float(len(dataNeutre))))) 
Neu=sorted(Neudata_list, key=lambda x:x[1],reverse=True)
print "******************************"
print "Neutre 20"
print "******************************"
for i in range(0,20):
     print Neu[i][0]



