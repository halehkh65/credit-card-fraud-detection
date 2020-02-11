import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import itertools


def main():
    # Loads the dataset
    printOptions()
    data = pd.read_csv("./dataset.csv")

    key = input("Select an option or enter 'q' for exit the program: ")
    while key.lower() != 'q':
        if int(key) == 1:
            infodata(data)
        elif int(key) == 2:
            outDetect(data)

        elif int(key) == 3:
            calculation(data)
        elif int(key) == 4:
            neuralNetwork(data)
        printOptions()
        key = input("Select an option or enter 'q' for exit the program: ")


def printOptions():
    print("1: Dataset information")
    print("2: Outlier detection:")

    print("3: Machine learning models and evaluations:")
    print("4: Neural network model:")


def infodata(df):

    #shows the dataset's statistical characteristics

    print("---Dataset Information---")
    print(df.info())
    print("*" * 100)
    print(df.head())
    print("*" * 100)
    print(df.describe())
    print("*" * 100)
    key = input("Would you like to see the Histogram of the dataset (y/n)? : ").lower()
    if key == 'y':
        histo(df)
    key = input("Would you like to see the Distribution of the dataset (y/n)? : ").lower()
    if key == 'y':
        dist(df)


def histo(df):
    #displays each column's histogram

    df.hist(figsize=(15, 15))
    plt.show()


def dist(data):
    #creates two plots to display "class","Amount" and "Time" distribution

    fraud_number = data.groupby('Class').size()[1]
    print("Fraud===", fraud_number)
    fraud_percent = (fraud_number / data['Class'].size) * 100
    print("Fraud Transactions Percentage in the Dataset is:", round(fraud_percent, 5), "%")
    nonfraud_number = data.groupby('Class').size()[0]
    nonfraud_percent = (nonfraud_number / data['Class'].size) * 100
    print("Valid Transactions Percentage in the Dataset is:", round(nonfraud_percent, 5), "%")
    print("*" * 100)
    # Plot distribution of "Class"

    colors = ["#000080", "#800000"]
    sns.countplot('Class', data=data, palette=colors)
    plt.title('Class Distributions \n (Fraud transactions vs valid transactions)', fontsize=14)
    plt.show()

    # Plot distribution of "Amount" and "Time" in the dataset

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 24))
    amount_val = data['Amount'].values
    time_val = data['Time'].values

    sns.distplot(amount_val, ax=ax1, color='r')
    ax1.set_title('Distribution of Transaction Amount', fontsize=14)
    ax1.set_xlim([min(amount_val), max(amount_val)])

    sns.distplot(time_val, ax=ax2, color='b')
    ax2.set_title('Distribution of Transaction Time', fontsize=14)
    ax2.set_xlim([min(time_val), max(time_val)])


def outDetect(data):

    #Creates correlation matrix to anomaly detection purposes
    #We will use boxplots to have a better understanding of the distribution of features in fraud and non-fraud transactions.

    f, ax = plt.subplots(1, 1, figsize=(24, 20))
    sub_sample_corr = data.corr()
    new_df = data.copy()
    sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size': 20}, ax=ax)
    ax.set_title('SubSample Correlation Matrix ', fontsize=14)


    f, axes = plt.subplots(ncols=3, figsize=(20, 4))

    sns.boxplot(x="Class", y="V14", data=new_df, ax=axes[0])
    axes[0].set_title('V17:Negative Correlation with label')

    sns.boxplot(x="Class", y="V12", data=new_df, ax=axes[1])
    axes[1].set_title('V14:Negative Correlation with label')

    sns.boxplot(x="Class", y="V10", data=new_df, ax=axes[2])
    axes[2].set_title('V12:Negative Correlation with label')

    f, axes = plt.subplots(ncols=3, figsize=(20, 4))
    sns.boxplot(x="Class", y="V2", data=new_df, ax=axes[0])
    axes[0].set_title('V11:Negative Correlation with label')

    sns.boxplot(x="Class", y="V4", data=new_df, ax=axes[1])
    axes[1].set_title('V2:Positive Correlation with label')

    sns.boxplot(x="Class", y="V11", data=new_df, ax=axes[2])
    axes[2].set_title('V19:Positive Correlation with label')
    plt.show()

def outRemove(data):
    """removes detected outliers: from boxplots we know that V14 and V10 are negatively correlated.
    we are using "Numeric outlier detection" technique.
    low outliers < q25−1.5⋅IQR
    high outliers > q75+1.5⋅IQR
    """
    new_df = data.copy()

    non_fraud_number = new_df.groupby('Class').size()[0]
    #  V14 outliers removing
    print("Number of non-frauds transactions before V14 outlier removing is:", non_fraud_number)
    fraud_number = new_df.groupby('Class').size()[1]
    print("Number of frauds transactions before V14 outlier removing is", fraud_number)
    print(new_df.info())
    v14_fraud = new_df[14].loc[new_df['Class'] == 1].values
    q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)
    v14_iqr = q75 - q25

    v14_cut_off = v14_iqr * 1.5
    v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off

    outliers = [x for x in v14_fraud if x < v14_lower or x > v14_upper]
    print("Number of V14 outliers is: {}".format(len(outliers)))
    removed_transactions = new_df[(new_df[14] > v14_upper) | (new_df[14] < v14_lower)]
    removed_non_frauds = removed_transactions[(removed_transactions["Class"] == 0)]
    removed_frauds = removed_transactions[(removed_transactions["Class"] == 1)]
    print("Number of removed non-frauds transactions  is:{}".format(len(removed_non_frauds)))
    print("Number of removed frauds transactions  is:{}".format(len(removed_frauds)))
    new_df = new_df.drop(new_df[(new_df[14] > v14_upper) | (new_df[14] < v14_lower)].index)

    print(100 * "*")


    # V10 outliers removing
    v10_fraud = new_df[10].loc[new_df['Class'] == 1].values
    q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)
    v10_iqr = q75 - q25
    v10_cut_off = v10_iqr * 1.5
    v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off
    outliers = [x for x in v10_fraud if x < v10_lower or x > v10_upper]
    print('Number of V10 outliers is: {}'.format(len(outliers)))
    removed_transactions = new_df[(new_df[10] > v10_upper) | (new_df[10] < v10_lower)]
    removed_non_frauds = removed_transactions[(removed_transactions["Class"] == 0)]
    removed_frauds = removed_transactions[(removed_transactions["Class"] == 1)]
    print("Number of removed non-frauds transactions  is:{}".format(len(removed_non_frauds)))
    print("Number of removed frauds transactions  is:{}".format(len(removed_frauds)))

    new_df = new_df.drop(new_df[(new_df[10] > v10_upper) | (new_df[10] < v10_lower)].index)

    print(100 * "*")


    print('Number of Instances after outliers removal: {}'.format(len(new_df)))

    print(100 * "*")

    return new_df


def undersam(scalled_df):
    """
    implements "Random Under Sampling" which basically consists
    of removing normal transactions in order to have a more balanced dataset
    """
    from collections import Counter
    from imblearn.under_sampling import RandomUnderSampler
    import numpy as np
    import warnings
    X = scalled_df.drop('Class', axis=1)
    y = scalled_df["Class"]

    print('Original dataset shape {}'.format(Counter(y)))
    sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)#split our training and test sets

    for train_index, test_index in sss.split(X, y):
        print("Train:", train_index, "Test:", test_index)
        original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
        original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]
    original_Xtrain = original_Xtrain.values# Turn the values into an array for feeding the classification algorithms.
    original_Xtest = original_Xtest.values
    original_ytrain = original_ytrain.values
    original_ytest = original_ytest.values
    train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
    test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)
    print('-' * 100)

    print('Label Distributions: \n')
    print(train_counts_label / len(original_ytrain))
    print(test_counts_label / len(original_ytest))

    sampling_strategy = 1
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
    X_train, y_train = rus.fit_resample(original_Xtrain, original_ytrain)

    print('Resampled dataset shape {}'.format(Counter(y_train)))
    classifiers = {
        "Logisitic Regression": LogisticRegression(),
        "K Nearest": KNeighborsClassifier(),
        "Support Vector Machine": SVC(),
        "Decision Tree Classifier": DecisionTreeClassifier(),
        "Random Forest Classifier": RandomForestClassifier()

    }
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import metrics
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score
    for i, (clfName, clf), in enumerate(classifiers.items()):
        warnings.filterwarnings("ignore", category=FutureWarning)

        training_score = cross_val_score(clf, X_train, y_train, cv=5)
        print("Classifiers: ", clf.__class__.__name__, "Has a training score of",
              round(training_score.mean(), 2) * 100,
              "% accuracy score")
        tunedClf = tunedCalssifiers(clf.__class__.__name__, X_train, y_train)
        training_score = cross_val_score(tunedClf, X_train, y_train, cv=5)

        print("Tuned Classifiers: ", clf.__class__.__name__, "Has a training score of",
              round(training_score.mean(), 2) * 100, "% accuracy score")
        yPred = tunedClf.predict(original_Xtest)
        cf = confusion_matrix(original_ytest, yPred)
        print(cf)
        precision, recall, thresholds = precision_recall_curve(original_ytest, yPred)
        auc = metrics.auc(recall, precision)
        print("AUC=", auc)

        print(classification_report(original_ytest, yPred))
        print("Final Accuracy of ", clf.__class__.__name__, " is: ", round(accuracy_score(original_ytest, yPred), 2) * 100,
              "%")
        print(100 * "*")





def oversam(scalled_df):
    """
       Implements "SMOTE OverSampling" which basically creates synthetic points from the fraud samples in order to
       reach an equal balance between the minority and majority class.
       We also create a sub-sample of non-fraud transactions which is consist of just 10% of all non-fraud transactions.

    """
    from collections import Counter
    import numpy as np


    X = scalled_df.drop('Class', axis=1)
    y = scalled_df["Class"]

    print('Original dataset shape {}'.format(Counter(y)))
    # cross validating and split our train and test dataset

    sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

    for train_index, test_index in sss.split(X, y):
        print("Train:", train_index, "Test:", test_index)
        original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
        original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

    # we are creating a dataset consists of 10% of 0 data with label 0 and whole of the data with label 1  from the original train dataset
    original_Xtrain["Class"] = original_ytrain
    train_df = original_Xtrain

    import matplotlib.pyplot as plt
    from yellowbrick.cluster import KElbowVisualizer
    from collections import Counter
    from imblearn.over_sampling import SMOTE
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans

    fraud_percentage = [10, 20, 30, 40, 50]
    fraud_df = train_df.loc[train_df['Class'] == 1]
    all_df = train_df.shape[0]
    print("Train dataset size is:", all_df)

    #calculate 10 percent of the original train dataset size
    pop_size = int(0.1 * all_df)
    print("sub set dataset size is:", pop_size)

    #take the number of non-fraud transactions in the original train dataset
    non_fraud_df = (train_df.loc[train_df['Class'] == 0])
    # groups all non-fraud transactions to 10 clusters using K-means clustering technique, k=10
    kmeans = KMeans(n_clusters=10).fit(non_fraud_df)
    pred = kmeans.predict(non_fraud_df)
    frame = non_fraud_df.copy()
    frame['cluster'] = pred
    clusters = []
    location = []#store location of clusters samples in the dataset
    count = [] #store number of non-frauds in each cluster
    for i in range(10): #k=10
        print(i)
        clusters.append(frame[(frame["cluster"] == i)].index.values.tolist())
        location.append(non_fraud_df.loc[clusters[i]])
        count.append(location[i].shape[0] / non_fraud_df.shape[0])


    sample_count = []
    temp = []
    for percent in fraud_percentage:
        for i in range(10):
            fraud_percent = (percent / 100)
            fraud_num = int(fraud_percent * pop_size)#required number of frauds
            non_fr_num = int((1 - fraud_percent) * pop_size)#required number of non_frauds
            sample_count.append(int(count[i] * non_fr_num))#calculate number of required samples from each cluster of non-frauds
            temp.append(location[i].sample(n = sample_count[i]))#take randomly required number of samples from each cluster

        final_non_fraud = pd.concat(temp)
        from collections import Counter
        from sklearn.svm import LinearSVC

        portion = round((fraud_num / non_fr_num), 2)
        train_df = pd.concat([fraud_df, final_non_fraud])

        temp_X_train = (train_df.drop('Class', axis=1)).values# Turn the values into an array for feeding the classification algorithms.

        temp_y_train = train_df["Class"].values


        sm = SMOTE(ratio=portion)

        X_train, y_train = sm.fit_sample(temp_X_train, temp_y_train)

        train_unique_label, train_counts_label = np.unique(y_train, return_counts=True)
        test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)
        print('-' * 100)
        print('Label Distributions: \n')
        print(train_counts_label / len(y_train))
        print(test_counts_label / len(original_ytest))

        print('Resampled dataset shape {}'.format(Counter(y_train)))

        classifiers = {
            "Logisitic Regression": LogisticRegression(),
            "K Nearest": KNeighborsClassifier(),
            "Support Vector Machine": SVC(),
            "Decision Tree Classifier": DecisionTreeClassifier(),
            "Random Forest Classifier": RandomForestClassifier()

        }
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve
        from sklearn import metrics
        import warnings
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import classification_report
        from sklearn.metrics import accuracy_score


        for i, (clfName, clf), in enumerate(classifiers.items()):
            warnings.filterwarnings("ignore", category=FutureWarning)

            training_score = cross_val_score(clf, X_train, y_train, cv=5)
            print("Classifiers: ", clf.__class__.__name__, "Has a training score of",
                  round(training_score.mean(), 2) * 100,
                  "% accuracy score")
            tunedClf = tunedCalssifiers(clf.__class__.__name__, X_train, y_train)
            training_score = cross_val_score(tunedClf, X_train, y_train, cv=5)

            print("Tuned Classifiers: ", clf.__class__.__name__, "Has a training score of",
                  round(training_score.mean(), 2) * 100, "% accuracy score")
            yPred = tunedClf.predict(original_Xtest)
            cf = confusion_matrix(original_ytest, yPred)
            print(cf)
            precision, recall, thresholds = precision_recall_curve(original_ytest, yPred)
            auc = round(metrics.auc(recall, precision), 2)
            print("AUC===", auc)

            print(classification_report(original_ytest, yPred))
            print("Final Accuracy of ", clf.__class__.__name__, " is: ",
                  round(accuracy_score(original_ytest, yPred), 2) * 100,
                  "%")
            print(100 * "*")




def scal(data):
    rob_scaler = RobustScaler()
    data['scaled_amount'] = rob_scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
    data['scaled_time'] = rob_scaler.fit_transform(data['Time'].values.reshape(-1, 1))
    scaled_amount = data['scaled_amount']
    scaled_time = data['scaled_time']
    data.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
    data.drop(['Time', 'Amount'], axis=1, inplace=True)
    data.insert(0, 'scaled_time', scaled_time)
    data.insert(29, 'scaled_amount', scaled_amount)
    print("Dataset is scaled now:")
    print(data.head())
    scalled_df = data
    return scalled_df


def calculation(data):
    printOptions()
    key = input("Would you like to remove outliers? y/n")

    if str(key) == "y":
        data = outRemove(data)


    # Before any calculation we need to scale "amount" and "time" columns to have an integrated dataset

    #scalled_df = scal(data)
    print("dataset Information after Scalling:", data.info)
    classifiers(data)


def tunedCalssifiers(calssifier, X_train, y_train):
    # Use GridSearchCV to find the best parameters.
    if calssifier == 'LogisticRegression':
        # Logistic Regression
        params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        gridModel = GridSearchCV(LogisticRegression(), params)

    elif calssifier == 'KNeighborsClassifier':
        params = {"n_neighbors": list(range(2, 5, 1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
        gridModel = GridSearchCV(KNeighborsClassifier(), params)
    elif calssifier == 'SVC':
        # Support Vector Classifier
        params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
        gridModel = GridSearchCV(SVC(), params)
    elif calssifier == 'DecisionTreeClassifier':
        # DecisionTree Classifier
        params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2, 4, 1)),
                  "min_samples_leaf": list(range(5, 7, 1))}
        gridModel = GridSearchCV(DecisionTreeClassifier(), params)
    elif calssifier == 'RandomForestClassifier':
        # RandomForest Classifier
        params = {'n_estimators': [50, 100], 'max_features': ['auto', 'sqrt', 'log2'], 'max_depth': [4, 5, 6, 7, 8],
                  'criterion': ['gini', 'entropy']}
        gridModel = GridSearchCV(RandomForestClassifier(), params)

    gridModel.fit(X_train, y_train)
    tunedModel = gridModel.best_estimator_
    return tunedModel


def classifiers(scalled_df):
    key = input("Which sampling technique would you like to apply on the  dataset? " + "\n"
                                                                                       "o:oversampling" + "\n" + "u:undersampling" + "\n").lower()
    if key == 'o':

        oversam(scalled_df)

    elif key == 'u':
        undersam(scalled_df)

main()