import sys
import os


# Data analysis
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks')

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB

# Custom libraries
import plotting
import mlearning


# Define sentiment columns
sentiments = ['Negative', 'Neutral', 'Positive']


def explore(data):
    '''Explore the provided data'''

    print('--- Available columns ---')
    print(data.columns)
    print('\n')
    print('--- First few lines of data ---')
    print(data.head(10))
    print('\n')
    print('--- Missing data ---')
    print(data.isnull().sum())
    print('\n')
    print('--- Duplicates ---')
    print(data[data.duplicated('Sentence')])
    print('Number of duplicates:', len(data[data.duplicated('Sentence')].index))
    print('\n')
    print('--- Duplicate sentiments ---')
    print(any(data.loc[:, sentiments].sum(axis=1) > 1))


def wrangle(data, target=sentiments):
    '''Prepare data for classification'''

    raw_len = len(data.index)

    # Drop duplicate data
    data.drop_duplicates('Sentence', inplace=True)

    # Reverse one-hot encoding to obtain single target vector
    data['Sentiment'] = data.loc[:, target].idxmax(axis=1)
    data.drop(target, axis=1, inplace=True)

    # Define length of sentences by splitting per space
    data['NumWords'] = data['Sentence'].str.split().str.len()

    # Remove special characters without meaning
    #

    print('{} instances of {} ({:.3f}%) remaining.'.format(
        len(data.index), raw_len,
        len(data.index) / raw_len * 100.0))


def characterize(data):
    '''Characterize the data visually'''

    # Bias in terms of sentiment
    plt.figure()
    plotting.save_figure(sns.countplot(x='Sentiment', data=data),
                         'sentiment_balance')

    # Overview of length of sentences
    plt.figure()
    plotting.save_figure(sns.distplot(data['NumWords']), 'num_words')

    # Overview of length of sentences per sentiment
    plt.figure()
    colors = plotting.palette()
    fig, axs = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    for sentiment, ax in zip(sentiments, axs):
        color = next(colors)
        p = sns.distplot(data.loc[data['Sentiment'] == sentiment, 'NumWords'],
                         bins=8, ax=ax, color=color,
                         hist=True, kde=False, norm_hist=True,
                         fit=stats.lognorm, fit_kws={'color': color})
        p.set_title(sentiment)
    # plt.legend()
    plotting.save_figure(plt.gcf(), 'num_words_sentiments')

    # Overview of length of sentences per sentiment
    plt.figure()
    colors = plotting.palette()
    for sentiment, ax in zip(sentiments, axs):
        sns.distplot(data.loc[data['Sentiment'] == sentiment, 'NumWords'],
                     bins=8, label=sentiment,
                     hist=False, kde=False, norm_hist=True,
                     fit=stats.lognorm, fit_kws={'color': next(colors)})
    plt.legend()
    plotting.save_figure(plt.gcf(), 'num_words_fits')


def bag_of_words(X_train, X_test, y_train, y_test):
    param_grid = {'model': [MultinomialNB, ComplementNB],
                  'stop_words': [None, 'english'],
                  'ngram_range': [(1, 1), (1, 2)],
                  'alpha': [0.001, 0.01, 0.1, 1.0]}

    # param_grid = {'model': [MultinomialNB],
    #               'stop_words': [None],
    #               'ngram_range': [(1, 1)],
    #               'alpha': [1.0]}
    results = mlearning.cross_validate(mlearning.bag_of_words, ParameterGrid(param_grid),
                                       X_train, y_train)

    for r in results:
        print('AS: {:.3f}, BAS: {:.3f}'.format(r['accuracy_score'],
                                               r['balanced_accuracy_score']),
              r['parameters'])

    for method in ['accuracy_score', 'balanced_accuracy_score']:
        optimum = mlearning.optimal_result(results, method)
        print('Optimal result: {} with\n'.format(optimum[method]), optimum['parameters'])


def tf_idf(X_train, X_test, y_train, y_test):
    param_grid = {'model': [MultinomialNB, ComplementNB],
                  'stop_words': [None, 'english'],
                  'ngram_range': [(1, 1), (1, 2)],
                  'alpha': [0.001, 0.01, 0.1, 1.0]}

    results = mlearning.cross_validate(mlearning.tf_idf, ParameterGrid(param_grid),
                                       X_train, y_train)

    for r in results:
        print('AS: {:.3f}, BAS: {:.3f}'.format(r['accuracy_score'],
                                               r['balanced_accuracy_score']),
              r['parameters'])

    for method in ['accuracy_score', 'balanced_accuracy_score']:
        optimum = mlearning.optimal_result(results, method)
        print('Optimal result: {} with\n'.format(optimum[method]), optimum['parameters'])


def main():
    # Setup plot output directory
    output_dir = './plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # import pandas as pd; data = pd.read_excel('data/sentences_with_sentiment.xlsx');
    data = pd.read_excel('data/sentences_with_sentiment.xlsx')

    # Data exploration, characterization and wrangling
    # explore() and characterize() are optional
    # explore(data)
    wrangle(data)
    # characterize(data)

    # Divide data into training and testing subsets. As the data is imbalanced
    # with respect to the sentiments, the split is stratified. Given the small
    # amount of data, valdiation will be performed in stratified folds of the
    # training subset.
    X_train, X_test, y_train, y_test = train_test_split(
        data['Sentence'], data['Sentiment'],
        test_size=3, stratify=data['Sentiment'])
    print('{} training instances versus {} testing instances'.format(
        len(X_train.index), len(X_test.index)))

    # print(y_train.value_counts())
    # print(y_test.value_counts())
    # print(y_train.groupby('Sentiment').count())
    # print(y_test.groupby('Sentiment').count())

    # Various NLP attempts to classify the sentiment. They are roughly ordered
    # with respect to their complexity, starting with the most simple approach.
    bag_of_words(X_train, X_test, y_train, y_test)
    tf_idf(X_train, X_test, y_train, y_test)

    return 0


if __name__ == '__main__':
    sys.exit(main())
