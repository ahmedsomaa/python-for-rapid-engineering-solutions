import numpy as np                                      # needed for arrays
import pandas as pd                                     # needed for data frames
from sklearn.svm import SVC                             # support vector algorithm
from sklearn.tree import export_graphviz                # cool graph
from sklearn.metrics import accuracy_score              # grade the results
from sklearn.linear_model import Perceptron             # perceptron algorithm
from sklearn.tree import DecisionTreeClassifier         # decision tree algorithm
from sklearn.preprocessing import StandardScaler        # standarize data
from sklearn.neighbors import KNeighborsClassifier      # knn algorithm
from sklearn.linear_model import LogisticRegression     # logistic regression algorithm
from sklearn.ensemble import RandomForestClassifier     # random forest algorithm
from sklearn.model_selection import train_test_split    # splits database

# parameters for each algorithm
DT_DEPTH = 5                                            # decision tree depth
RF_TREES = 5                                            # random forest trees
LR_C_VAL = .25                                          # logistic regression c value
SVM_C_VAL = .25                                         # support vector machine c value
KNN_NEIGHBORS = 5                                       # k-nearest neighbors neighs
PPN_MAX_ITERATIONS = 7                                  # perceptron max iterations

# csv data file name
FILE_NAME = 'heart1.csv'

# features index
FEATURES_END = 13                                       # end index for the features
FEATURES_START = 0                                      # start index of the features

##################################################################################
# print_method_header:: print method name                                        #
##################################################################################
def print_method_header(num, method):
    print('\n----------------------------------------------')
    print(f'[Classifier {num}] {method}')
    print('----------------------------------------------')

##################################################################################
# print_analysis_results:: print the results of analysis                         #
##################################################################################
def print_analysis_results(test_sam, test_miss, test_acc, combined_sam, combined_miss, combined_acc):
    print('\nNumber in test: ', test_sam)
    print('Misclassified samples: %d' % test_miss)
    print('Accuracy: %.2f' % test_acc)
    print('Number in combined: ', combined_sam)
    print('Misclassified combined samples: %d' % combined_miss)
    print('Combined Accuracy: %.2f' % combined_acc)

##################################################################################
# perceptron:: perform analysis using perceptron                                 #
##################################################################################
def perceptron(x_trn_std, x_tst_std, y_trn, y_tst, iterations):
    # create the classifier
    ppn = Perceptron(max_iter=iterations, tol=1e-3, eta0=0.001,
                     fit_intercept=True, random_state=0, verbose=True)
    ppn.fit(x_trn_std, y_trn)                   # do the training

    y_pred = ppn.predict(x_tst_std)             # now try with the test data
    test_acc = accuracy_score(y_tst, y_pred)

    # combine the train and test data
    X_combined_std = np.vstack((x_trn_std, x_tst_std))
    y_combined = np.hstack((y_trn, y_tst))

    # we did the stack so we can see how the combination of test and train data did
    y_combined_pred = ppn.predict(X_combined_std)
    combined_samples = (y_combined != y_combined_pred).sum()
    combined_acc = accuracy_score(y_combined, y_combined_pred)

    # print analysis result
    print_analysis_results(len(y_tst), (y_tst != y_pred).sum(
    ), test_acc, len(y_combined), combined_samples, combined_acc)

    return test_acc, y_pred

##################################################################################
# logistic_regression:: perform analysis using logistic regression               #
##################################################################################
def logistic_regression(x_trn_std, x_tst_std, y_trn, y_tst, c_val):
    # create the classifier
    lr = LogisticRegression(C=c_val, solver='liblinear',
                            multi_class='ovr', random_state=0)
    lr.fit(x_trn_std, y_trn)                    # do the training

    y_pred = lr.predict(x_tst_std)              # now try with the test data
    test_acc = accuracy_score(y_tst, y_pred)

    # combine the train and test data
    X_combined_std = np.vstack((x_trn_std, x_tst_std))
    y_combined = np.hstack((y_trn, y_tst))

    # we did the stack so we can see how the combination of test and train data did
    y_combined_pred = lr.predict(X_combined_std)
    combined_samples = (y_combined != y_combined_pred).sum()
    combined_acc = accuracy_score(y_combined, y_combined_pred)

    # print analysis result
    print_analysis_results(len(y_tst), (y_tst != y_pred).sum(
    ), test_acc, len(y_combined), combined_samples, combined_acc)

    return test_acc, y_pred

##################################################################################
# support_vector_machine:: perform analysis using support vector machine         #
##################################################################################
def support_vector_machine(x_trn_std, x_tst_std, y_trn, y_tst, c_val):
    # create the classifier
    svm = SVC(kernel='linear', C=c_val, random_state=0)
    svm.fit(x_trn_std, y_trn)                   # do the training

    y_pred = svm.predict(x_tst_std)             # now try with the test data
    test_acc = accuracy_score(y_tst, y_pred)

    # combine the train and test data
    X_combined_std = np.vstack((x_trn_std, x_tst_std))
    y_combined = np.hstack((y_trn, y_tst))

    # we did the stack so we can see how the combination of test and train data did
    y_combined_pred = svm.predict(X_combined_std)
    combined_samples = (y_combined != y_combined_pred).sum()
    combined_acc = accuracy_score(y_combined, y_combined_pred)

    # print analysis result
    print_analysis_results(len(y_tst), (y_tst != y_pred).sum(
    ), test_acc, len(y_combined), combined_samples, combined_acc)
    
    return test_acc, y_pred

##################################################################################
# decision_tree:: perform analysis using decision tree                           #
##################################################################################
def decision_tree(x_trn, x_tst, y_trn, y_tst, depth, cols):
    # create the classifier
    tree = DecisionTreeClassifier(
        criterion='entropy', max_depth=depth, random_state=0)
    tree.fit(x_trn, y_trn)                      # do the training

    y_pred = tree.predict(x_tst)                # now try with test data
    test_acc = accuracy_score(y_tst, y_pred)

    # combine the train and test data
    X_combined = np.vstack((x_trn, x_tst))
    y_combined = np.hstack((y_trn, y_tst))

    # we did the stack so we can see how the combination of test and train data did
    y_combined_pred = tree.predict(X_combined)
    combined_samples = (y_combined != y_combined_pred).sum()
    combined_acc = accuracy_score(y_combined, y_combined_pred)

    # print analysis result
    print_analysis_results(len(y_tst), (y_tst != y_pred).sum(
    ), test_acc, len(y_combined), combined_samples, combined_acc)

    # export the file tree.dot. To view this file
    # NOTE: you may have to install first...
    # Then execute: dot -T png -O tree.dot
    # Then execute: open tree.dot.png
    export_graphviz(tree, out_file='tree.dot', feature_names=cols)

    return test_acc, y_pred

##################################################################################
# random_forest:: perform analysis using random forest                           #
##################################################################################
def random_forest(x_trn, x_tst, y_trn, y_tst, trees):
    # create the classifier
    forest = RandomForestClassifier(
        criterion='entropy', n_estimators=trees, random_state=1, n_jobs=4)
    forest.fit(x_trn, y_trn)                    # do the training

    y_pred = forest.predict(x_tst)              # try with the test data
    test_acc = accuracy_score(y_tst, y_pred)

    # combine the train and test data
    X_combined = np.vstack((x_trn, x_tst))
    y_combined = np.hstack((y_trn, y_tst))

    # we did the stack so we can see how the combination of test and train data did
    y_combined_pred = forest.predict(X_combined)
    combined_samples = (y_combined != y_combined_pred).sum()
    combined_acc = accuracy_score(y_combined, y_combined_pred)

    # print analysis result
    print_analysis_results(len(y_tst), (y_tst != y_pred).sum(
    ), test_acc, len(y_combined), combined_samples, combined_acc)

    return test_acc, y_pred

##################################################################################
# k_nearest:: perform analysis using k-nearest neigbhors                         #
##################################################################################
def k_nearest(x_trn_std, x_tst_std, y_trn, y_tst, neighs):
    # create the classifier
    knn = KNeighborsClassifier(n_neighbors=neighs, p=2, metric='minkowski')
    knn.fit(x_trn_std, y_trn)                   # do the training

    y_pred = knn.predict(x_tst_std)             # now try with the test data
    test_acc = accuracy_score(y_tst, y_pred)

    # combine the train and test data
    X_combined_std = np.vstack((x_trn_std, x_tst_std))
    y_combined = np.hstack((y_trn, y_tst))

    # we did the stack so we can see how the combination of test and train data did
    y_combined_pred = knn.predict(X_combined_std)
    combined_samples = (y_combined != y_combined_pred).sum()
    combined_acc = accuracy_score(y_combined, y_combined_pred)

    # print analysis result
    print_analysis_results(len(y_tst), (y_tst != y_pred).sum(
    ), test_acc, len(y_combined), combined_samples, combined_acc)

    return test_acc, y_pred

##################################################################################
# main:: project entry point                                                     #
##################################################################################
def main():
    # load the data set
    df = pd.read_csv(FILE_NAME)
    print(f'[info]: {FILE_NAME} was successfully read')

    # convert data frame to numpy array
    numpy_df = df.to_numpy()

    # split the data to training & testing datasets
    X = numpy_df[:,:FEATURES_END]               # separate all the features
    y = numpy_df[:, FEATURES_END].ravel()       # extract the classifications

    # split the problem into train and test: 70% training and 30% test
    # random_state allows the split to be reproduced
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    # mean and standard deviation may be overridden with options
    sc = StandardScaler()                       # create the standard scalar
    sc.fit(X_train)                             # compute the required transformation
    X_train_std = sc.transform(X_train)         # apply to the training data
    X_test_std = sc.transform(X_test)           # and SAME transformation of test data

    # ---------------------------------- Perceptron
    print_method_header(1, 'perceptron')
    ppn_acc, ppn_pred = perceptron(X_train_std, X_test_std, y_train, y_test, PPN_MAX_ITERATIONS)

    # ---------------------------------- Logistic Regression
    print_method_header(2, 'logistic regression')
    lr_acc, lr_pred = logistic_regression(X_train_std, X_test_std, y_train, y_test, LR_C_VAL)

    # ---------------------------------- Support Vector Machine
    print_method_header(3, 'support vector machine')
    svm_acc, svm_pred = support_vector_machine(X_train_std, X_test_std, y_train, y_test, SVM_C_VAL)

    # ---------------------------------- Decision Tree
    print_method_header(4, 'decision tree')
    dt_acc, dt_pred = decision_tree(X_train, X_test, y_train, y_test, DT_DEPTH, 
    df.columns.values[FEATURES_START:FEATURES_END])

    # ---------------------------------- Random Forest
    print_method_header(5, 'random forest')
    rf_acc, rf_pred = random_forest(X_train, X_test, y_train, y_test, RF_TREES)

    # ---------------------------------- K-Nearest Neighbors
    print_method_header(6, 'k-nearest neighbors')
    knn_acc, knn_pred = k_nearest(X_train, X_test, y_train, y_test, KNN_NEIGHBORS)

    # ---------------------------------- Ensemble Learning Part
    print('\n----------------------------------------------')
    print(f'Classifiers Accuracies')
    print('----------------------------------------------')
    print('Perceptron Accuracy:', round(ppn_acc, 2))
    print('Logistic Regression Accuracy:', round(lr_acc, 2))
    print('Support Vector Machine Accuracy:', round(svm_acc, 2))
    print('Decision Tree Accuracy:', round(dt_acc, 2))
    print('Random Forest Accuracy:', round(rf_acc, 2))
    print('K-Nearest Neighbors Accuracy:', round(knn_acc, 2))

    # add classifiers accuracy & y_pred in a tuple list
    classifiers = [(ppn_acc, ppn_pred), (lr_acc, lr_pred), (svm_acc, svm_pred), 
    (dt_acc, dt_pred), (rf_acc, rf_pred), (knn_acc, knn_pred)]

    # sort list by highest accuracy
    classifiers.sort(key= lambda a: a[0], reverse=True)

    # ---------------------------------- Ensemble Learning with 3
    threshold = 2.5 # threshold for 3 methods
    sum_of_methods = classifiers[0][1] + classifiers[1][1] + classifiers[2][1]  # add the highest thress methods
    ensemble3_pred = np.where(sum_of_methods > threshold, 2, 1) # do the predicition
    ensemble3_accu = accuracy_score(y_test, ensemble3_pred) # find the score
    print("Ensemble with Three Methods Accuracy:", round(ensemble3_accu, 2)) # print the score


# call main
main()
