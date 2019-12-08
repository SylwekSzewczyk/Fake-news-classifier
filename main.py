# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 15:09:09 2019

@author: Sylwek Szewczyk
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

class FakeNews:
    
    def __init__(self, db):
        self.db = db
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.classifier = None
        self.tfidf_vectorizer = None
        
    def showData(self):
        print(self.db.shape)
        print(self.db.head())
    
    def splitData(self, testsize):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.db.text, self.db.label, test_size = testsize, random_state = 7)
    
    def solve(self):
        
        self.tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df=0.7)
        tfidf_train = self.tfidf_vectorizer.fit_transform(self.x_train)
        tfidf_test = self.tfidf_vectorizer.transform(self.x_test)
        self.classifier = PassiveAggressiveClassifier(max_iter = 50)
        self.classifier.fit(tfidf_train, self.y_train)
        y_pred = self.classifier.predict(tfidf_test)
        score = accuracy_score(self.y_test, y_pred)
        print('Accuracy of the solved model: {}'.format(round(100*score,2)))
        cm = confusion_matrix(self.y_test, y_pred, labels = ['FAKE', 'REAL'])
        print(cm)
    
    def predict(self, news):
        return self.classifier.predict(self.tfidf_vectorizer.transform([news]))
    
    @classmethod
    def loadData(cls, data):
        return cls(db=pd.read_csv(data))

f = FakeNews.loadData('news.csv')
f.splitData(0.3)
f.solve()

f.predict('Donald Trump has resigned from being the president of the United States of America')
