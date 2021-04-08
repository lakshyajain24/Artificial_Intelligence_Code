#Importing Libraries
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import sklearn
import pickle


#Preprocesing the Input
def Prepare_text(text):
    text = re.sub('[^a-zA-Z]',' ',text)
    text = text.lower()
    text = text.split()
    review = [word for word in text if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review ]
    review = " ".join(review)
    user = []
    user.append(review)
    return user


def sentiment_predict(textData):
    file = open("pickle_model.pkl","rb")
    Predicter = pickle.load(file)
    vocab=pickle.load(open("feature.pkl","rb"))
    
    from sklearn.feature_extraction.text import CountVectorizer
    recreated_vec = CountVectorizer(decode_error = 'replace', vocabulary = vocab)
    
    
    return Predicter.predict(recreated_vec.fit_transform(textData))


def Conditions_Check(sentext):
    if(sentext==[0]):
        review ="Negative"
    else:
        review ="Positive"
    return(review)
    

#Taking Input
text = input("Enter The Text>>")

sent = sentiment_predict(Prepare_text(text))

Conditions_Check(sent)