import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC, SVC

data = pd.read_csv("C:/Users/karme/Downloads/spy_data.csv",index_col='Date')


def model_variables(prices, lags):

    # Change data types of prices dataframe from object to numeric
    prices = prices.apply(pd.to_numeric)
    # Create the new lagged DataFrame
    inputs = pd.DataFrame(index=prices.index)

    inputs["Close"] = prices["Close"]
    inputs["Volume"] = prices["Volume"]
    # Create the shifted lag series of prior trading period close values
    for i in range(0, lags):
        tsret = pd.DataFrame(index=inputs.index)
        inputs["Lag%s" % str(i + 1)] = prices["Close"].shift(i + 1)

    # Create the returns DataFrame
    tsret["VolumeChange"] = inputs["Volume"].pct_change()
    tsret["returns"] = inputs["Close"].pct_change() * 100.0

    # If any of the values of percentage returns equal zero, set them to
    # a small number (stops issues with QDA model in Scikit-Learn)
    for i, x in enumerate(tsret["returns"]):
        if abs(x) < 0.0001:
            tsret["returns"][i] = 0.0001

    # Create the lagged percentage returns columns
    for i in range(0, lags):
        tsret["Lag%s" % str(i + 1)] = \
            inputs["Lag%s" % str(i + 1)].pct_change() * 100.0

    # Create the "Direction" column (+1 or -1) indicating an up/down day
    tsret = tsret.dropna()
    tsret["Direction"] = np.sign(tsret["returns"])

    # Convert index to datetime in order to filter the dataframe by dates when 
    # we create the train and test dataset
    tsret.index = pd.to_datetime(tsret.index)
    return tsret


# Pass the dataset(data) and the number of lags 2 as the inputs of the model_variables  function
variables_data = model_variables(data, 2)

# Use the prior two days of returns and the volume change as predictors
# values, with direction as the response
dataset = variables_data[["Lag1", "Lag2", "VolumeChange", "Direction"]]
dataset = dataset.dropna()

# Create the dataset with independent variables (X) and dependent variable y
X = dataset[["Lag1", "Lag2", "VolumeChange"]]
y = dataset["Direction"]

# Split the train and test dataset using the date in the date_split variable
# This will create a train dataset of 4 years data and a test dataset for more than 
# 9 months data.

date_split = datetime.datetime(2019, 1, 1)

X_train = X[X.index <= date_split]
X_test = X[X.index > date_split]
y_train = y[y.index <= date_split]
y_test = y[y.index > date_split]

# Create the (parametrised) models
print("Hit Rates/Confusion Matrices:\n")
models = [("LR", LogisticRegression()),
          ("LDA", LDA()),
          ("LSVC", LinearSVC()),
          ("RSVM", SVC(
              C=1000000.0, cache_size=200, class_weight=None,
              coef0=0.0, degree=3, gamma=0.0001, kernel='rbf',
              max_iter=-1, probability=False, random_state=None,
              shrinking=True, tol=0.001, verbose=False)
           ),
          ("RF", RandomForestClassifier(
              n_estimators=1000, criterion='gini',
              max_depth=None, min_samples_split=2,
              min_samples_leaf=30, max_features='auto',
              bootstrap=True, oob_score=False, n_jobs=1,
              random_state=None, verbose=0)
           )]

# Iterate through the models and obtain the accuracy metrix: Hit Rate and Confusion Matrix
for m in models:

    m[1].fit(X_train, y_train)

    pred = m[1].predict(X_test)


    print("%s:\n%0.3f" % (m[0], m[1].score(X_test, y_test)))
    print("%s\n" % confusion_matrix(pred, y_test))