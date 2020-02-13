# credit-card-fraud-detection
Python
---

Author: "Haleh Taghdisi"

Date: '01/10/19'

---

## Performance Analysis of Machine Learning (ML) Algorithms on Highly Imbalanced Credit Card Datasets:
Five ML algorithms (Logistic Regression, k- Nearest Neighbors, Support Vector Machine, Decision Tree, and Random Forest) trained with a dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud), which contains transaction information made by a series of European cardholders during two days in September 2013. 

Results are reported regarding two representative metrics: `Recall` and `AUC Precision-Recall`.

Resampling methods ( Random Under Sampling  (RUS) and  Synthetic Minority Oversampling Technique (SMOTE) ) are implemented to add or remove examples from the training dataset in order to change the class distribution.

When oversampling is chosen as the resampling method, the balanced dataset size would be incredibly large, so we tried to find an optimized proportion of frauds to non-frauds in the training dataset to obtain a decision boundary in the feature space for each ML algorithm.

We have applied two popular techniques to shrink this massive amount of non-fraud samples and provide a meaningful subset:

1. Sampling: taking a uniform selection of the entirety of non-fraud transactions and creating a subsequence of them.
2. Clustering: usingthe k-means method to clustering non-fraud samples and taking a representative sub-sample of each cluster. to reduce the number of non-frauds transactions.


Results show that ML algorithms are classified into two groups. First, ML algorithms uniformly perform competently: kNN and RF. Results show these algorithms after training with a properly augmentation of minority class and sampling of the majority class detect more frauds without pointing many non-frauds as a fraud. Second, ML algorithms have inconsistent behavior: SVM, DT and LR. As such, SVM had
improvement in AUC when genius and synthetic fraud samples are up to 10% of the train dataset. DT had recall and AUC extension for 10-25% fraud samples and decreased for higher portions. The LR uniformly had recall improvement, but AUC was mostly decreased.

This project code is implemented in PyCharm 2017.3.3 Professional Edition.

For running we need:

    • Python 3.6 (Anaconda3)
    • pandas
    • numpy
    • tensorflow
    • seaborn
    • sklearn
    • itertools
    • matplotlib
    • Dataset which is dataset.csv file

