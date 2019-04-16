import sys
import os


# Data analysis
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks')


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

    print('\n')
    print('{} instances of {} ({:.3f}%) remaining.'.format(
        len(data.index), raw_len,
        len(data.index) / raw_len * 100.0))


def main():
    # Setup plot output directory
    output_dir = './plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # import pandas as pd; data = pd.read_excel('data/sentences_with_sentiment.xlsx');
    data = pd.read_excel('data/sentences_with_sentiment.xlsx')

    explore(data)
    wrangle(data)

    return 0


if __name__ == '__main__':
    sys.exit(main())
