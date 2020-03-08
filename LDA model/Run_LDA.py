import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.decomposition import LatentDirichletAllocation



df = pd.read_csv('movies_labeled.csv',encoding = "ISO-8859-1")


print("PREPROCESSING DATA...")
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_df = 0.95, min_df = 2,stop_words = 'english')
X_vec = cv.fit_transform(df['Content'])
print("PREPROCESSING COMPLETE")


print("TRAINING...")
lda = LatentDirichletAllocation(n_components = 25,max_iter=60, random_state = 42)
lda.fit(X_vec)
print("TRAINING COMPLETE")




for index, topic in enumerate(lda.components_):
    print(f'Top 5 words for Topic #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-5:]])





X = []
for i in range(0,2000):
        feature = lda.transform(X_vec[i])
        flat_feature = feature.tolist()
        X.append(flat_feature[0])

"""
Create label list + set hyperparameters for the SVM classifier
"""

list_of_ones = [1]*1000
list_of_zeroes = [0]*1000
Y = list_of_ones + list_of_zeroes

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


clf = svm.SVC(kernel='linear')
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
scores = cross_val_score(clf, X, Y, cv=cv,scoring='f1')
print(np.mean(scores))
