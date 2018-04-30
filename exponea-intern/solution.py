import argparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

def predict_class_prob(model, x_test):
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:,1]

    return y_pred, y_prob

def scale_cont_data(data):
    # normalize continuous variables
    scaler = StandardScaler()
    data[:,:10] = scaler.fit(data[:,:10]).transform(data[:,:10])

    return data

def show_summary(model, x_test, y_test):
    y_pred, y_prob = predict_class_prob(model, x_test)

    # print report
    model.report = classification_report(y_test, y_pred, target_names=["no","yes"])
    print(model.report)
    accuracy = accuracy_score(y_test, y_pred)
    conf = confusion_matrix(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)
    print("Accuracy: {0:.2f}%".format(100*accuracy))
    print("Confusion table: {}".format(conf))
    print("AUC score: {0:.2f}%".format(100*auc_score))

def logit_preprocess(data):
    categ_labels = ['international plan', 'voice mail plan', 'class', 'number customer service calls']

    data_categ = data.loc[:,categ_labels]
    data_cont = data.drop(categ_labels, 1)

    dummy_customer = pd.get_dummies(data_categ['number customer service calls'], prefix='service calls')
    data_categ.drop('number customer service calls', 1, inplace=True)
    data_categ = data_categ.join(dummy_customer.loc[:,'service calls_1':])
    # service calls_0 is left as default

    data_logit = data_cont.join(data_categ)
    X = data_logit.drop("class", 1).values
    Y = data_logit["class"]

    # splitting 50:50
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=123)

    X_train_logit = scale_cont_data(X_train)
    X_test_logit = scale_cont_data(X_test)

    return X_train_logit, X_test_logit, Y_train, Y_test

##########################################
# LOADING
print("Loading files...")

# process two files
parser = argparse.ArgumentParser()
parser.add_argument("--file", "-f", nargs=2, required=True)
args = parser.parse_args()

with open(args.file[1], 'r') as f:
    labels = [row.split(":")[0] for row in f.readlines() if ":" in row]
    labels.append("class")

data = pd.read_csv(args.file[0], names=labels)

##########################################
# CLEANING
print("Checking for invalid data")

nulldata = len(np.where(pd.isnull(data))[0])
baddata = len(np.where(data.applymap(lambda x: x in ['', ' ', None]))[0])

if (nulldata & baddata == 0):
    print("Data are clean! (no Nones and empty strings)")

print("Cleaning data")

# fixing formats of binary valued features
data['international plan'] = data['international plan'].map({' yes': 1, ' no': 0})
data['voice mail plan'] = data['voice mail plan'].map({' yes': 1, ' no': 0})
data['class'] = data['class'].map({' False.': 0, ' True.': 1})

# dropping unuseful features - states, area codes, phone numbers
data.drop(['state', 'area code', 'phone number'], 1, inplace=True)


# dropping the second corr partners
data.drop(['number vmail messages',
           'total day charge',
           'total eve charge',
           'total night charge',
           'total intl charge'], 1, inplace=True)

# updating labels
labels = list(data)
labels_features = labels[:-1]

##########################################
# LOGISTIC REGRESSION
print("Logistic regression preprocessing..")

X_train_logit, X_test_logit, Y_train, Y_test = logit_preprocess(data)

C_range = [0.01,0.1,1,10,100,1000] # regularizatio
penalty_range = ['l1', 'l2'] # ridge and lasso method

param_grid = {'C': C_range,
              'penalty': penalty_range}

grid_search = GridSearchCV(LogisticRegression(fit_intercept = True,
                                              class_weight="balanced"),
                           param_grid=param_grid,
                           scoring="f1")

grid_search.fit(X_train_logit, Y_train)
print("Best logit parameters: {}".format(grid_search.best_params_))

# choose the final best model
logit = grid_search.best_estimator_
logit.fit(X_train_logit, Y_train)
show_summary(logit, X_test_logit, Y_test)

#########################################
# DECISION TREE
print("Decision tree preprocessing..")

X = data.drop("class", 1).values
Y = data["class"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=123)

# testing DT with various parameters
minleaf_range = [1,2,3,5]
minsplit_range = [2,5,10,15,20]
maxdepth_range = [2,3,5,8,10]

param_grid = {'min_samples_leaf': minleaf_range,
              'min_samples_split': minsplit_range,
              'max_depth': maxdepth_range}

grid_search = GridSearchCV(DecisionTreeClassifier(criterion = "entropy",
                                                  random_state = 100),
                           param_grid=param_grid, scoring="f1")

grid_search.fit(X_train, Y_train)
print("Best DT parameters: {}".format(grid_search.best_params_))

# choose the final best model
tree = grid_search.best_estimator_
tree.fit(X_train, Y_train)
show_summary(tree, X_test, Y_test)

#########################################
# DECISION TREE
print("Random forest preprocessing..")

# testing RF with various parameters
minleaf_range = [1,2,5]
minsplit_range = [2,5,8,10]
maxdepth_range = [5,10,12,15]
boot_range = [True, False]

param_grid = {'min_samples_leaf': minleaf_range,
              'min_samples_split': minsplit_range,
              'max_depth': maxdepth_range,
              'bootstrap': boot_range}

grid_search = GridSearchCV(RandomForestClassifier(criterion = "entropy",
                                                  random_state = 100,
                                                  n_estimators=100),
                           param_grid=param_grid,
                           scoring="accuracy",
                           n_jobs = -1)

# print the best params
grid_search.fit(X_train, Y_train)
print("Best RF parameters: {}".format(grid_search.best_params_))

# choose the final best model
forest = grid_search.best_estimator_
forest.fit(X_train, Y_train)
show_summary(forest, X_test, Y_test)

print("Finished !")
