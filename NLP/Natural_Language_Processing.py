import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

#nltk.download('wordnet')
#nltk.download('punkt')

data=pd.read_csv("tweets.csv",encoding="utf8")
nd=data
data.drop(["Handle"],axis=1,inplace=True)
data=pd.concat([data.head(10000),data.tail(10000)],axis=0,ignore_index=True)
data.Party=[1 if i=="Democrat" else 0 for i in data.Party]

data.Tweet=pd.DataFrame([i.split('https')[0]] for i in data.Tweet)

descriptions=[]

lemma=nltk.WordNetLemmatizer()

for tweet in data.Tweet:
    tweet=re.sub("[^a-zA-Z]"," ",tweet)
    tweet.lower()
    tweet=nltk.word_tokenize(tweet)
    tweet=[lemma.lemmatize(i) for i in tweet]
    tweet=" ".join(tweet)
    descriptions.append(tweet)

count_vectorizer = CountVectorizer(max_features=4000,stop_words="english")

x=count_vectorizer.fit_transform(descriptions).toarray()
y=data.iloc[:,0].values

x,x_test,y,y_test=train_test_split(x,y,test_size=0.1,random_state=3)

nb=GaussianNB()
nb.fit(x,y)

print("Accuracy=",nb.score(nb.predict(x_test).reshape(-1,1),y_test))

