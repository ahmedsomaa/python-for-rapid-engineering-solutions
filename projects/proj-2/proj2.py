import numpy as np                                      # for arrays and not a number
import pandas as pd                                     # for data frames
import matplotlib.pyplot as plt                         # for plotting
from warnings import filterwarnings                     # for warnings
from sklearn.decomposition import PCA                   # PCA package
from sklearn.metrics import accuracy_score              # grade the results
from sklearn.metrics import confusion_matrix            # generate the matrix
from sklearn.neural_network import MLPClassifier        # Multilayer Perceptron
from sklearn.preprocessing import StandardScaler        # scaling data
from sklearn.model_selection import train_test_split    # splits database

# csv data file name
FILE_NAME = 'sonar_all_data_2.csv'

# features index
N_COMPONENTS = 62               # number of file components
N_FEATURES = N_COMPONENTS - 2   # number of features to use
RANGE_END = N_FEATURES + 1      # add one to include feature 60 in range

##################################################################################
# print_line:: print the following +------------+----------+ to screen           #
##################################################################################
def print_line():
    print('+------------+----------+')

##################################################################################
# pca_analysis:: perform PCA analysis using MLPClassifier                        #
##################################################################################
def pca_analysis(x_trn_sd, x_tst_sd, y_trn, y_tst):
    acc = []        # initial accuracy list
    confuse = []    # initial confusion matrix
    # loop on all components
    print_line()
    print('| Components | Accuracy |')
    print_line()
    for n_comps in range(1, RANGE_END):
        # apply principal component analysis
        pca = PCA(n_components=n_comps)
        X_train_pca = pca.fit_transform(x_trn_sd)    # apply to the train data
        X_test_pca = pca.transform(x_tst_sd)  # do the same to the test data

        # now create a multilayer perceptron and train on it
        mlp = MLPClassifier(hidden_layer_sizes=(200, 100), activation='relu',
                            max_iter=200, alpha=.0001, solver='adam',  tol=.001, 
                            learning_rate='constant', random_state=1)
        mlp.fit(X_train_pca, y_trn)                   # do the training

        y_pred = mlp.predict(X_test_pca)    # now try with the test data
        accuracy = accuracy_score(y_tst, y_pred)

        # append accuracy with n_comp to scores list
        acc.append(accuracy)

        # append to confusion matrix
        confuse.append(confusion_matrix(y_tst, y_pred))

        # print #of components and accuracy
        
        print(f'| {n_comps: <{10}} | {round(accuracy, 2): <{8}} |')
        print_line()
        # print(f'n_components = {n_comps}, test accuracy = {round(accuracy, 2)}')
    return acc, confuse

##################################################################################
# plot_accuracy_vs_ncomps:: plot accuracies vs number of componenets             #
##################################################################################
def plot_accuracy_vs_ncopms(n_comps, scores):
    plt.plot(n_comps, scores, label='Test Accuracy', marker='x')
    plt.xlabel('Number of Components')
    plt.ylabel('Test Accuracy')
    plt.title('Machine Learning Mine Versus Rock')
    plt.legend(loc='lower right')
    plt.show()

##################################################################################
# main:: project entry point                                                     #
##################################################################################
def main():
    # read data file
    df = pd.read_csv(FILE_NAME, header=None)
    print(f'[info]: {FILE_NAME} was successfully read\n')

    # split the data to training & testing datasets
    X = df.iloc[:, :N_FEATURES].values     # features are from 0:59
    y = df.iloc[:,  N_FEATURES].values     # classes are in column 60

    # split the problem into train and test: 70% training and 30% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    # ---------------------------------- Standardization
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    # ---------------------------------- PCA
    # pca_analysis returns accuracy list and confusion matrix
    acc_list, conf_mtx = pca_analysis(X_train_std, X_test_std, y_train, y_test)

    # compute max accuracy and #of components to achieve it
    max_accuracy = np.max(acc_list)
    n_components = acc_list.index(max_accuracy) + 1

    # print maximum accuracy with #of components
    print(f'\nmax accuracy = {round(max_accuracy, 2)} was achieved with n_components = {n_components}')

    # plot accuracy against n_comps
    components = range(1, RANGE_END, 1)
    plot_accuracy_vs_ncopms(components, acc_list)

    # ---------------------------------- Confusion Matrix
    # create and print the confusion matrix
    print('confusion matrix for n_components =', n_components)
    print(conf_mtx[n_components - 1])

# ignore warnings
filterwarnings('ignore')

# call main
main()
