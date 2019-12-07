# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 15:09:09 2019

@author: Sylwek Szewczyk
"""

import numpy as np
import itertools
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score

class FakeNews:
    
    def __init__(self, db):
        self.db = db
    
    def showData(self):
        print(self.db.shape)
        return self.db.head()
    
    @classmethod
    def loadData(cls, data):
        return cls(db=pd.read_csv(data))

f = FakeNews.loadData('news.csv')
x = f.db
print(x)