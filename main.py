import sys
import os
import functools


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


# Define constant analysis parameters
sentiments = ['Negative', 'Neutral', 'Positive']
scores = ['confusion_matrix',
          'accuracy_score', 'accuracy_score_std',
          'balanced_accuracy_score', 'balanced_accuracy_score_std']
selected_score = 'balanced_accuracy_score'
selected_score_std = selected_score + '_std'



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

    # Pie chart to visualize fraction of useful data
    print('{} instances of {} ({:.3f}%) remaining.'.format(
        len(data.index), raw_len,
        len(data.index) / raw_len * 100.0))
    plt.figure()
    labels = ['Good data', 'Bad data']
    sizes = [len(data.index), raw_len - len(data.index)]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.tight_layout()
    plotting.save_figure(plt.gcf(), 'useful_data')


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


def split(data):
    '''Splits the data into training and testing subsets

    As the data is imbalanced with respect to the sentiments, the splitting is
    stratified. The goal is to operate with the same relative amounts of
    sentiments in order to avoid testing with samples consistint predominantly
    of one category.'''

    return train_test_split(data['Sentence'], data['Sentiment'],
        test_size=0.25, stratify=data['Sentiment'], random_state=0)


def conjunction(conditions):
    return functools.reduce(np.logical_and, conditions)


def evaluate_parameter(results, poi, categorical=False, log=False, xticklabels=None):
    # Replace NaN (or None) with 0 to allow for Python-style logical comparison
    results = results.fillna(0)
    # Create parameter index (without scores) to filter results with
    optimum = results.iloc[results[selected_score].idxmax()]
    idx = optimum.index.drop(scores)
    # Get sample of results where all parameters are identical except for the
    # parameter of interest
    sample = results[conjunction([results[n] == optimum[n] for n in idx.drop(poi)])]
    print(sample)

    if categorical:
        sns.catplot(x=poi, y=selected_score, data=sample, kind='bar')
        # if xticklabels:
        #     plt.xticklabels(xticklabels)
        plt.ylabel(selected_score)
        plt.xlabel(poi)
    else:
        plt.figure()
        plt.errorbar(x=poi, y=selected_score, data=sample)
        plt.fill_between(sample[poi],
                        sample[selected_score] - sample[selected_score_std],
                        sample[selected_score] + sample[selected_score_std],
                        alpha=0.2)
        plt.ylabel(selected_score)
        plt.xlabel(poi)
        if log: plt.xscale('log')
        plotting.save_figure(plt.gcf(), 'alpha')


def bag_of_words(data):
    param_grid = {'model': [MultinomialNB, ComplementNB, BernoulliNB],
                  'stop_words': [None, 'english'],
                  'ngram_range': [(1, 1), (1, 2)],
                  'min_df': [1, 2, 3],
                  'alpha': [0.001, 0.01, 0.1, 1.0]}
    # More parameters to test:
    # max_df, max_features


    # param_grid = {'model': [MultinomialNB, ComplementNB, BernoulliNB],
    #               'stop_words': [None],
    #               'ngram_range': [(1, 1)],
    #               'min_df': [1],
    #               'alpha': [0.01, 0.1]}


    results = mlearning.cross_validate(mlearning.bag_of_words,
                                       ParameterGrid(param_grid),
                                       data['Sentence'], data['Sentiment'])
    print('--- Bag-of-words results ---')
    print(results
          .sort_values(by='balanced_accuracy_score', ascending=False)
          .loc[:, ['balanced_accuracy_score', 'balanced_accuracy_score_std', 'model']]
          .head(10))

    evaluate_parameter(results, 'alpha', False, True)
    evaluate_parameter(
        results, 'model', categorical=True, xticklabels=
        [str(m).split("'")[1].split('.')[2] for m in param_grid['model']])



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
    explore(data)
    wrangle(data)
    characterize(data)

    # Various NLP attempts to classify the sentiment. They are roughly ordered
    # with respect to their complexity, starting with the most simple approach.
    bag_of_words(data)
    # tf_idf(X_train, X_test, y_train, y_test)



    return 0


if __name__ == '__main__':
    sys.exit(main())
