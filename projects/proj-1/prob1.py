import numpy as np                                  # for arrays and not a number
import pandas as pd                                 # for data frames
import seaborn as sns                               # for data visualization
import matplotlib.pyplot as plt                     # for plotting

TOP = 10                                            # top 10 for corr & cov
FILE_NAME = 'heart1.csv'                            # name of file to read
VARIABLE_TO_PREDICT = 'a1p2'                        # variable to predict

##################################################################################
# correlation:: return correlation matrix and highly correlated variables        #
##################################################################################
def correlation(df):
    # create the correlation matrix
    # take the absolute value since large negative are as useful as large positive
    corr = df.corr().abs()
    print('[Table 1]: correlation of each variable with all other variables')
    print(corr)

    # clear redundant values, since correlation with itself is always 1
    corr *= np.tri(*corr.values.shape, k=-1).T
    
    # stack values so they can be sorted
    corr_unstack = corr.unstack()

    # sort values in descending order to get the variables with highest correlation
    corr_unstack.sort_values(inplace=True, ascending=False)

    # return high correlated variables, 
    # and also the highly correlated variables with {VARIABLE_TO_PREDICT}
    return corr_unstack.head(TOP), corr_unstack[VARIABLE_TO_PREDICT].head(TOP)


##################################################################################
# covariance:: create covariance matrix and return top covariances               #
##################################################################################
def covariance(df):
    # create the covariance matrix
    # take the absolute value since large negative are as useful as large positive
    cov = df.cov().abs()
    print('\n[Table 4]: covariance of each variable with all other variables\n')
    print(cov)

    # clear redundant values, since correlation with itself is always 1
    cov *= np.tri(*cov.values.shape, k=-1).T

    # stack values so they can be sorted
    cov_unstack = cov.unstack()

    # sort values in descending order to get the variables with highest correlation
    cov_unstack.sort_values(inplace=True, ascending=False)

    # return high variables with high covariance,
    # and also the variables with high covariance with {VARIABLE_TO_PREDICT}
    return cov_unstack.head(TOP), cov_unstack[VARIABLE_TO_PREDICT].head(TOP)

##################################################################################
# has_null_values:: checks if a data frame has null values                       #
##################################################################################
def has_null_values(df):
    return df.isnull().values.sum() == 0

##################################################################################
# draw_pair_plot:: creates a pair plot                                           #
##################################################################################
def draw_pair_plot(df):
    sns.set(style='whitegrid', context='notebook')
    sns.pairplot(df, height=2.5)
    plt.show()

##################################################################################
# main:: project entry point                                                     #
##################################################################################
def main():
    # read data file
    df = pd.read_csv(FILE_NAME)
    print(f'[info]: {FILE_NAME} was successfully read')

    # check for null values or missing data
    if has_null_values(df):
        print('[info]: no need for imputing as data has no missing nor null values\n')
    else:
        print('[error]: imputing is needed as data has missing or nul values\n')
    
    # print correlation & get highly correlated variables
    highly_corr, highly_corr_predict = correlation(df)
    print('\n[Table 2]: higly correlated variables')
    print(highly_corr)

    # print higly correlated variables with {VARIABLE_TO_PREDICT}
    print(f'\n[Table 3]: higly correlated variables with "{VARIABLE_TO_PREDICT}" variable\n')
    print(highly_corr_predict)

    # print covariance & get variables with high covariance
    highly_cov, highly_cov_predict = covariance(df)
    print('\n[Table 5]: variables with high covariance\n')
    print(highly_cov)

    # print variables with high covariance against {VARIABLE_TO_PREDICT}
    print(f'\n[Table 6]: variables with high covariance against {VARIABLE_TO_PREDICT}\n')
    print(highly_cov_predict)

    # create the pair plot
    draw_pair_plot(df)

# call project entry poitn
main()
