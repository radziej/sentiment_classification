import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix


def score(y_true, y_pred):
    sentiments = ['Negative', 'Neutral', 'Positive']
    score = {
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=sentiments),
        'accuracy_score': accuracy_score(y_true, y_pred),
        'balanced_accuracy_score': balanced_accuracy_score(y_true, y_pred)
    }
    return score


def average_scores(results):
    result = {
        'confusion_matrix': np.mean([r['confusion_matrix'] for r in results], axis=0),
        'accuracy_score': np.mean([r['accuracy_score'] for r in results]),
        'accuracy_score_std': np.std([r['accuracy_score'] for r in results]),
        'balanced_accuracy_score': np.mean([r['balanced_accuracy_score'] for r in results]),
        'balanced_accuracy_score_std': np.std([r['balanced_accuracy_score'] for r in results])
    }
    return result


def cross_validate(procedure, parameter_grid, X, y, k=30):
    results = []
    for parameters in parameter_grid:
        # For the given set of parameters, determine the average performance
        intermediates = []
        skf = StratifiedKFold(n_splits=k, random_state=0)
        for train_idx, validate_idx in skf.split(X, y):
            # Split samples into training and validation subsets
            X_train, X_val = X.iloc[train_idx], X.iloc[validate_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[validate_idx]
            model, y_pred = procedure(parameters, X_train, X_val, y_train)
            intermediates.append(score(y_val, y_pred))
        # Store average scores and parameters for evaluation against test data
        result = average_scores(intermediates)
        result.update(parameters)
        results.append(result)

    # Convert list of dictionaries to DataFrame for easier processing
    return pd.DataFrame(results)


def bag_of_words(parameters, X_train, X_val, y_train):
    vec = CountVectorizer(stop_words=parameters['stop_words'],
                          ngram_range=parameters['ngram_range'],
                          min_df=parameters['min_df'])
    vec.fit(X_train, y_train)
    X_train = vec.transform(X_train)
    X_val = vec.transform(X_val)

    model = parameters['model'](alpha=parameters['alpha'])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    return model, y_pred


def tf_idf(parameters, X_train, X_val, y_train):
    vec = TfidfVectorizer(stop_words=parameters['stop_words'],
                          ngram_range=parameters['ngram_range'])
    vec.fit(X_train, y_train)
    X_train = vec.transform(X_train)
    X_val = vec.transform(X_val)

    model = parameters['model'](alpha=parameters['alpha'])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    return model, y_pred
