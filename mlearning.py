import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.ensemble import ExtraTreesClassifier

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.utils import tokenize
from gensim.models import Word2Vec
from nltk.stem import SnowballStemmer

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix

sentiments = ['Negative', 'Neutral', 'Positive']

def score(y_true, y_pred):
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


def characterize_optimum(results, data, procedure):
    optimum = results.loc[results['balanced_accuracy_score'].idxmax()]
    # Stratified train-test split to minimize skewed sentiment distribution
    X_train, X_test, y_train, y_test = train_test_split(
        data['Sentence'], data['Sentiment'],
        test_size=0.25, stratify=data['Sentiment'], random_state=0)
    model, y_pred = procedure(optimum['parameters'],
                              X_train, X_test, y_train, characterize=True)


def cross_validate(procedure, parameter_grid, X, y, k=5, cache=True):
    # Reads cache if requested
    output_path = os.path.join('cache', procedure.__name__ + '.pkl')
    if cache and os.path.exists(output_path):
        return pd.read_pickle(output_path)

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
        result['parameters'] = parameters
        results.append(result)

    # Convert list of dictionaries to DataFrame for an easier estimation of the
    # performance for different parameters and cache it to skip recalculating
    # it (unless requested)
    df = pd.DataFrame(results)
    df.to_pickle(output_path)
    return df


def filter_parameters(parameters, tag, divider='__'):
    return {k.split(divider, 1)[1]: v
            for k, v in parameters.items()
            if k.startswith(tag + divider)}


# Vectorizer classes built on top of the default sklearn options to allow for
# stemming of words
english_stemmer = SnowballStemmer('english')
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([english_stemmer.stem(w) for w in analyzer(doc)])


class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: ([english_stemmer.stem(w) for w in analyzer(doc)])



def bag_of_words(parameters, X_train, X_val, y_train, characterize=False):
    vec = StemmedCountVectorizer(**filter_parameters(parameters, 'vec'))
    # vec = CountVectorizer(**filter_parameters(parameters, 'vec'))
    vec.fit(X_train, y_train)
    X_train = vec.transform(X_train)
    X_val = vec.transform(X_val)

    model = parameters['model'](**filter_parameters(parameters, 'clf'))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    if characterize:
        characterize_term_frequency_matrix(X_train, y_train, vec.get_feature_names())

    return model, y_pred


def tf_idf(parameters, X_train, X_val, y_train, characterize=False):
    vec = StemmedTfidfVectorizer(**filter_parameters(parameters, 'vec'))
    vec.fit(X_train, y_train)
    X_train = vec.transform(X_train)
    X_val = vec.transform(X_val)

    model = parameters['model'](**filter_parameters(parameters, 'clf'))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    if characterize:
        characterize_term_frequency_matrix(X_train, y_train, vec.get_feature_names())

    return model, y_pred

def characterize_term_frequency_matrix(X, y, columns):
    df = pd.DataFrame(X.toarray(), columns=columns)
    df['Sentiment'] = y
    for sentiment in sentiments:
        print('--- {} ---'.format(sentiment))
        print(df[df['Sentiment'] == sentiment]
              .sum()
              .drop('Sentiment')
              .sort_values(ascending=False).head(20))

    # Determine feature importance via decision tree classifier
    etc = ExtraTreesClassifier(n_estimators=100)
    etc.fit(X, y)
    df = pd.Series(etc.feature_importances_, index=columns)
    print(df.sort_values(ascending=False).head(30))


def latent_semantic_analysis(parameters, X_train, X_val, y_train, characterize=True):
    vec = StemmedTfidfVectorizer(**filter_parameters(parameters, 'vec'))
    vec.fit(X_train, y_train)
    X_train = vec.transform(X_train)
    X_val = vec.transform(X_val)

    lsa = TruncatedSVD(**filter_parameters(parameters, 'lda'))
    lsa.fit(X_train, y_train)
    X_train = lsa.transform(X_train)
    X_val = lsa.transform(X_val)

    if characterize:
        characterize_lsa_components(lsa, vec.get_feature_names())

    model = parameters['model'](**filter_parameters(parameters, 'clf'))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    return model, y_pred


def characterize_lsa_components(lsa, columns, n_components=3):
    df = pd.DataFrame(lsa.components_, columns=columns)
    print(df[df.abs().sort_values(by=10, axis=1, ascending=False).columns]
          .iloc[10:10 + n_components, 0:7].to_latex(
              float_format=lambda x: '{:.3f}'.format(x)))


def latent_dirichlet_allocation(parameters, X_train, X_val, y_train, characterize=False):
    vec = StemmedTfidfVectorizer(**filter_parameters(parameters, 'vec'))

    vec.fit(X_train, y_train)
    X_train = vec.transform(X_train)
    X_val = vec.transform(X_val)

    lda = LatentDirichletAllocation(**filter_parameters(parameters, 'lda'))
    lda.fit(X_train, y_train)
    X_train = lda.transform(X_train)
    X_val = lda.transform(X_val)

    if characterize:
        characterize_topics(lda, vec.get_feature_names())

    model = parameters['model'](**filter_parameters(parameters, 'clf'))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    return model, y_pred


def nonnegative_matrix_factorization(parameters, X_train, X_val, y_train, characterize=False):
    vec = StemmedTfidfVectorizer(**filter_parameters(parameters, 'vec'))

    vec.fit(X_train, y_train)
    X_train = vec.transform(X_train)
    X_val = vec.transform(X_val)

    nmf = NMF(random_state=0, **filter_parameters(parameters, 'nmf'))
    nmf.fit(X_train, y_train)
    X_train = nmf.transform(X_train)
    X_val = nmf.transform(X_val)

    if characterize:
        characterize_topics(nmf, vec.get_feature_names())

    model = parameters['model'](**filter_parameters(parameters, 'clf'))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    return model, y_pred


def characterize_topics(model, columns):
    def get_top_words(model, words, n_top_words=6):
        topic_content = []
        for i, topic in enumerate(model.components_):
            topic_content.append([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        return topic_content

    for i, words in enumerate(get_top_words(model, columns)):
        print('{} & {} \\\\'.format(i + 1, ', '.join(words)))
    print()


def word2vec(parameters, X_train, X_val, y_train, characterize=False):
    def convert(series):
        tokens = []
        for i in range(len(series)):
            tokens.append(list(tokenize(series.iloc[i], **filter_parameters(parameters, 'tok'))))
        return tokens

    def featurize(model, sentences):
        '''Create Word2Vec features for sentences

        Sentences are already assumed'''

        features = np.zeros((len(sentences), model.vector_size))
        for i, sentence in enumerate(sentences):
            for word in sentence:
                try:
                    features[i,:] = model[word]
                except KeyError:
                    continue
        return features

    X_train = convert(X_train)
    X_val = convert(X_val)

    w2v = Word2Vec(X_train, size=100, window=5, min_count=1, workers=8)
    w2v.train(X_train, total_examples=len(X_train), epochs=15)

    if characterize:
        print(w2v['sufficient'])

    X_train = featurize(w2v, X_train)
    X_val = featurize(w2v, X_val)

    model = parameters['model'](**filter_parameters(parameters, 'clf'))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    return model, y_pred
