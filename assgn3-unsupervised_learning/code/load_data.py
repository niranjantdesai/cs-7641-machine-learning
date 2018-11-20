import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_breast_cancer_data(filename):
    """
    Loads the Breast Cancer Wisconsin (Diagnostic) Data Set
    :param filename: path to csv file
    :return: X (data) and y (labels)
    """

    data = pd.read_csv(filename)

    # y includes our labels and x includes our features
    y = data.diagnosis  # malignant (M) or benign (B)
    # The column "Unnamed: 32" feature includes NaN so drop it from the data. Also drop "id" as it is not a feature and
    # "diagnosis" as it is the label
    to_drop = ['Unnamed: 32', 'id', 'diagnosis']
    X = data.drop(to_drop, axis=1)

    # Convert string labels to numerical values
    y = y.values
    y[y == 'M'] = 1
    y[y == 'B'] = 0
    y = y.astype(int)

    return X, y


def load_mushroom_data(filename):
    """
    Loads the Mushroom Classification Data Set
    :param filename: path to csv file
    :return: X (data) and y (labels)
    """

    data = pd.read_csv(filename)

    # y includes our labels and x includes our features
    X = data.iloc[:, 1:23]
    y = data.iloc[:, 0]

    # As the feature values are in strings, perform label encoding to convert all the unique values to integers
    labelencoder = LabelEncoder()
    for col in X.columns:
        X[col] = labelencoder.fit_transform(X[col])

    # Convert string labels to numerical values
    y = y.values
    y[y == 'p'] = 1
    y[y == 'e'] = 0
    y = y.astype(int)

    return X, y

def load_wine_quality_data(filename):
    """
    Loads the wine quality dataset
    :param filename: path to csv file
    :return: X (data) and y (labels)
    """

    data = pd.read_csv(filename, sep=';')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    y = y.values
    y = y.astype(int)
    y[y < 6] = 0
    y[y >= 6] = 1

    return X, y

def load_wine_quality_data_orig(filename):
    """
    Loads the wine quality dataset
    :param filename: path to csv file
    :return: X (data) and y (labels)
    """

    data = pd.read_csv(filename, sep=';')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    y = y.values
    y = y.astype(int)

    return X, y
