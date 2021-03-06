import sys
import os
import functools


# Data analysis
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks')
sns.set_context('talk')

# Machine learning
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Custom libraries
import plotting
import mlearning


# Define constant analysis parameters
sentiments = ['Negative', 'Neutral', 'Positive']
scores = ['confusion_matrix',
          'accuracy_score', 'accuracy_score_std',
          'balanced_accuracy_score', 'balanced_accuracy_score_std',
          'parameters']
selected_score = 'balanced_accuracy_score'
selected_score_std = selected_score + '_std'


def setup():
    def ensure_path(path):
        if not os.path.exists(path):
            os.makedirs(path)

    ensure_path('data')
    ensure_path('plots')
    ensure_path('cache')


def explore(data):
    '''Explore the provided data'''

    print('--- Available columns ---')
    print(data.columns)
    print()
    print('--- First few lines of data ---')
    print(data.head(10))
    print()
    print('--- Missing data ---')
    print(data.isnull().sum())
    print()
    print('--- Duplicates ---')
    print(data[data.duplicated('Sentence')])
    print('Number of duplicates:', len(data[data.duplicated('Sentence')].index))
    print()
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

    # Visualize splitting of data
    plt.figure()
    fold = StratifiedKFold(n_splits=5)
    plotting.cross_validation_indices(
        fold, data['Sentence'],
        data['Sentiment'].map(
            {'Negative': 0, 'Neutral': 1, 'Positive': 2}).sort_values(),
    ax, 5)
    plotting.save_figure(plt.gcf(), 'stratified_cross_fold')


def conjunction(conditions):
    return functools.reduce(np.logical_and, conditions)


def evaluate_parameter(results, poi, tag,
                       xlabel=None, ylabel=None,
                       categorical=False, log=False, xticklabels=None):
    # Replace NaN (or None) with 0 to allow for Python-style logical comparison
    results = results.fillna(0)
    # Create parameter index (without scores) to filter results with
    optimum = results.loc[results[selected_score].idxmax()]
    idx = optimum.index.drop(scores)
    # Get sample of results where all parameters are identical except for the
    # parameter of interest
    sample = results[conjunction([results[n] == optimum[n] for n in idx.drop(poi)])]

    if categorical:
        c = sns.catplot(x=poi, y=selected_score, data=sample, kind='bar')
        plt.ylabel(ylabel if ylabel else selected_score)
        plt.xlabel(xlabel if xlabel else poi)
        if xticklabels:
            vals, labels = plt.xticks()
            plt.xticks(vals,
                       [xticklabels(l.get_text()) for l in labels],
                       rotation=30)
            plt.xlabel('')
        plotting.save_figure(plt.gcf(), tag + '_' + poi)
    else:
        sample = sample.sort_values(by=poi)
        plt.figure()
        plt.errorbar(x=poi, y=selected_score, data=sample)
        plt.fill_between(sample[poi],
                        sample[selected_score] - sample[selected_score_std],
                        sample[selected_score] + sample[selected_score_std],
                        alpha=0.2)
        plt.ylabel(ylabel if ylabel else selected_score)
        plt.xlabel(xlabel if xlabel else poi)
        if log: plt.xscale('log')
        plotting.save_figure(plt.gcf(), tag + '_' + poi)


def bag_of_words(data):
    param_grid = [
        {'vec__stop_words': [None, 'english'],
         'vec__ngram_range': [(1, 1), (1, 2)],
         'vec__min_df': [3, 4, 5],
         'vec__max_df': [0.9, 0.95],
         'model': [LinearSVC],
         'clf__max_iter': [10000],
         'clf__dual': [False, True],
         'clf__class_weight': ['balanced'],
         'clf__random_state': [0]},

        {'vec__stop_words': [None, 'english'],
         'vec__ngram_range': [(1, 1), (1, 2)],
         'vec__min_df': [1, 2, 3, 4],
         'vec__max_df': [0.5, 0.75, 0.9, 0.95],
         'model': [MultinomialNB, ComplementNB, BernoulliNB],
         'clf__alpha': [0.001, 0.01, 0.1, 1.0]}
    ]

    results = mlearning.cross_validate(
        mlearning.bag_of_words, ParameterGrid(param_grid),
        data['Sentence'], data['Sentiment']
    ).sort_values(by='balanced_accuracy_score', ascending=False)
    print('--- Bag-of-words results ---')
    print(results
          .drop(['accuracy_score', 'accuracy_score_std', 'confusion_matrix', 'parameters'], axis=1)
          .rename({'balanced_accuracy_score': 'bas', 'balanced_accuracy_score_std': 'bas_std'}, axis=1)
          .head(20))

    mlearning.characterize_optimum(results, data, mlearning.bag_of_words)
    # results = results[pd.notna(results['clf__alpha'])]
    # evaluate_parameter(results, 'clf__alpha', 'bow', False, True)
    # evaluate_parameter(
    #     results, 'model', 'bow', categorical=True, xticklabels=
    #     [str(m).split("'")[1].split('.')[-1] for m in results['model'].unique()])


def tf_idf(data):
    param_grid = [
        {'vec__stop_words': [None, 'english'],
         'vec__ngram_range': [(1, 1), (1, 2)],
         'vec__min_df': [3, 4, 5],
         'vec__max_df': [0.5, 0.75, 0.9],
         # 'vec__sublinear_tf': [True, False],
         'model': [LinearSVC],
         'clf__max_iter': [10000],
         'clf__dual': [False, True],
         'clf__class_weight': ['balanced'],
         'clf__random_state': [0]},

        {'vec__stop_words': [None, 'english'],
         'vec__ngram_range': [(1, 1), (1, 2)],
         'vec__min_df': [1, 2, 3, 4],
         'vec__max_df': [0.5, 0.75, 0.9, 0.95],
         'vec__sublinear_tf': [True, False],
         'model': [MultinomialNB, ComplementNB, BernoulliNB],
         'clf__alpha': [0.001, 0.01, 0.1, 1.0]}
    ]

    results = mlearning.cross_validate(
        mlearning.tf_idf, ParameterGrid(param_grid),
        data['Sentence'], data['Sentiment']
    ).sort_values(by='balanced_accuracy_score', ascending=False)
    print('--- Tf-Idf results ---')
    print(results
          .drop(['accuracy_score', 'accuracy_score_std', 'confusion_matrix', 'parameters'], axis=1)
          .rename({'balanced_accuracy_score': 'bas', 'balanced_accuracy_score_std': 'bas_std'}, axis=1)
          .head(20))

    mlearning.characterize_optimum(results, data, mlearning.tf_idf)

    # Confusion matrix
    optimum = results.loc[results['balanced_accuracy_score'].idxmax()]
    plotting.confusion_heatmap(optimum['confusion_matrix'], sentiments)
    plotting.save_figure(plt.gcf(), 'confusion_heatmap')
    plotting.confusion_heatmap(optimum['confusion_matrix'], sentiments, True)
    plotting.save_figure(plt.gcf(), 'confusion_heatmap_norm')

    # Hyperparameter dependency
    evaluate_parameter(results, 'clf__alpha', 'tfidf',
                       ylabel='Balanced accuracy score', xlabel='Alpha', log=True)
    evaluate_parameter(results, 'vec__max_df', 'tfidf',
                       ylabel='Balanced accuracy score', xlabel='Max doc frequency')
    evaluate_parameter(results, 'vec__min_df', 'tfidf',
                       ylabel='Balanced accuracy score', xlabel='Min doc frequency')
    evaluate_parameter(results, 'vec__ngram_range', 'tfidf',
                       ylabel='Balanced accuracy score', xlabel='n-gram range', categorical=True)
    evaluate_parameter(
        results, 'model', 'tfidf', ylabel='Balanced accuracy score', categorical=True,
        xticklabels=lambda x: str(x).split("'")[1].split('.')[-1])


def latent_semantic_analysis(data):
    param_grid = [
        {'vec__stop_words': [None, 'english'],
         'vec__ngram_range': [(1, 1), (1, 2)],
         'vec__min_df': [1, 2],
         'vec__max_df': [0.5, 0.75, 0.9, 0.95],
         'vec__sublinear_tf': [True, False],
         'lda__n_components': [10, 20, 30],
         'model': [RandomForestClassifier],
         'clf__random_state': [0],
         'clf__n_estimators': [150, 200, 300, 500],
         'clf__max_depth': [10, 15, None]},

        # Not close enough
        # {'vec__stop_words': [None, 'english'],
        #  'vec__ngram_range': [(1, 1), (1, 2)],
        #  'vec__min_df': [1, 2],
        #  'lda__n_components': [30, 50, 70],
        #  'model': [LinearSVC],
        #  'clf__max_iter': [10000],
        #  'clf__dual': [False, True],
        #  'clf__class_weight': ['balanced'],
        #  'clf__random_state': [0]},

        # Does not even come close
        # {'vec__stop_words': [None, 'english'],
        #  'vec__ngram_range': [(1, 1), (1, 2)],
        #  'vec__min_df': [1, 2],
        #  'lda__n_components': [30, 50, 70],
        #  'model': [LogisticRegression],
        #  'clf__multi_class': ['multinomial'],
        #  'clf__solver': ['newton-cg']}
    ]

    results = mlearning.cross_validate(
        mlearning.latent_semantic_analysis, ParameterGrid(param_grid),
        data['Sentence'], data['Sentiment'])
    print('--- LSA results ---')
    print(results
          .sort_values(by='balanced_accuracy_score', ascending=False)
          .drop(['accuracy_score', 'accuracy_score_std', 'confusion_matrix'], axis=1)
          .rename({'balanced_accuracy_score': 'bas', 'balanced_accuracy_score_std': 'bas_std'}, axis=1)
          .head(10))
    mlearning.characterize_optimum(results, data, mlearning.latent_semantic_analysis)


def latent_dirichlet_allocation(data):
    param_grid = {
        'vec__stop_words': [None, 'english'],
        'vec__ngram_range': [(1, 1)],  # (1, 2) doesnt perform well
        'vec__min_df': [1, 2],
        'vec__max_df': [0.4, 0.5, 0.6],
        'lda__n_components': [10, 20, 30],
        'lda__learning_offset': [10., 20., 40.],
        'lda__random_state': [0],
        'model': [RandomForestClassifier],
        'clf__random_state': [0],
        'clf__n_estimators': [100, 150, 200],
        'clf__max_depth': [10, 15, None]
    }

    results = mlearning.cross_validate(
        mlearning.latent_dirichlet_allocation, ParameterGrid(param_grid),
        data['Sentence'], data['Sentiment'])
    print('--- LDA results ---')
    print(results
          .sort_values(by='balanced_accuracy_score', ascending=False)
          .drop(['accuracy_score', 'accuracy_score_std', 'confusion_matrix', 'parameters'], axis=1)
          .rename({'balanced_accuracy_score': 'bas', 'balanced_accuracy_score_std': 'bas_std'}, axis=1)
          .head(10))

    mlearning.characterize_optimum(results, data, mlearning.latent_dirichlet_allocation)

def nonnegative_matrix_factorization(data):
    param_grid = {
        'vec__stop_words': [None, 'english'],
        'vec__ngram_range': [(1, 1), (1, 2)],
        'vec__min_df': [1, 2],
        'vec__max_df': [0.5, 0.75, 0.9, 0.95],
        'nmf__n_components': [10, 20, 30],
        'nmf__beta_loss': ['kullback-leibler'],
        'nmf__solver': ['mu'],
        'nmf__max_iter': [1000],
        'nmf__alpha': [0.0, 0.1],
        # 'nmf__l1_ratio': [0.5],
        'model': [RandomForestClassifier],
        'clf__random_state': [0],
        'clf__n_estimators': [150, 200, 300, 500],
        'clf__max_depth': [10, 15, None]
    }

    results = mlearning.cross_validate(
        mlearning.nonnegative_matrix_factorization, ParameterGrid(param_grid),
        data['Sentence'], data['Sentiment'])
    print('--- NMF results ---')
    print(results
          .sort_values(by='balanced_accuracy_score', ascending=False)
          .drop(['accuracy_score', 'accuracy_score_std', 'confusion_matrix', 'parameters'], axis=1)
          .rename({'balanced_accuracy_score': 'bas', 'balanced_accuracy_score_std': 'bas_std'}, axis=1)
          .head(10))
    mlearning.characterize_optimum(results, data, mlearning.nonnegative_matrix_factorization)


def word2vec(data):
    param_grid = {
        'tok__lowercase': [True],
        'tok__deacc': [True],
        'model': [RandomForestClassifier],
        'clf__random_state': [0],
        'clf__n_estimators': [100, 150, 200, 300, 500],
        'clf__max_depth': [10, 15, None]
    }

    results = mlearning.cross_validate(
        mlearning.word2vec, ParameterGrid(param_grid),
        data['Sentence'], data['Sentiment'])
    print('--- W2V results ---')
    print(results
          .sort_values(by='balanced_accuracy_score', ascending=False)
          .drop(['accuracy_score', 'accuracy_score_std', 'confusion_matrix'], axis=1)
          .rename({'balanced_accuracy_score': 'bas', 'balanced_accuracy_score_std': 'bas_std'}, axis=1)
          .head(10))
    mlearning.characterize_optimum(results, data, mlearning.word2vec)


def main():
    # Setup directories
    setup()
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
    tf_idf(data)
    latent_semantic_analysis(data)
    latent_dirichlet_allocation(data)
    nonnegative_matrix_factorization(data)
    word2vec(data)

    return 0


if __name__ == '__main__':
    sys.exit(main())
