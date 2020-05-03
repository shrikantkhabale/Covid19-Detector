import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pickle

def data_split(data,ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data)*ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

if __name__ == "__main__":
    # read the data from csv
    df = pd.read_csv('dummycoviddata.csv')
    train,test = data_split(df,0.2)

    X_train = train.drop('infectionProb',axis=1).to_numpy()
    X_test = test.drop('infectionProb',axis=1).to_numpy()
    Y_train = train['infectionProb'].to_numpy()
    Y_test = test['infectionProb'].to_numpy()


    clf = LogisticRegression()
    #clf = BernoulliNB()
    #clf = KNeighborsClassifier()
    #clf = SVC(gamma='auto')

    clf.fit(X_train,Y_train)

    # open file where you want to save data
    file = open('model.pkl','wb')
    # dump information to that file
    pickle.dump(clf,file)

    file.close()
   
    
    #y_pred = clf.predict(X_test)

    #clf.predict_proba(X_test)

    #print(f"Number of mislabeled points out of a total {X_test.shape[0]} points : {(Y_test != y_pred).sum()}")