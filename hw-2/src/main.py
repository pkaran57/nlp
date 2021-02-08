import logging
import os
import random

import gensim.downloader as api
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD

from definitions import OUTPUT_DIR

np.random.seed(0)
random.seed(0)

logging.basicConfig(format="'%(asctime)s' %(name)s : %(message)s'", level=logging.INFO)
logger = logging.getLogger('main')


def load_embedding_model(model_name):
    """
    Load word2vec vectors
    :return: wv_from_bin: All embeddings, each length 300
    """
    # this downloads the 300-dimensional word2vec vectors trained on Google News corpus, more here: https://radimrehurek.com/gensim/models/word2vec.html
    wv_from_bin = api.load(model_name)
    logger.info("Loaded vocab of size {}".format(len(wv_from_bin.vocab.keys())))
    return wv_from_bin


def get_matrix_of_vectors(wv_from_bin, required_words=['portland', 'pacific', 'forest', 'ocean', 'city', 'food', 'green', 'weird', 'cool', 'oregon']):
    """ Put the word2vec vectors into a matrix M.
        Param:
            wv_from_bin: KeyedVectors object; the 3000000 word2vec vectors loaded from file
        Return:
            M: numpy matrix shape (num words, 300) containing the vectors
            word2ind: dictionary mapping each word to its row number in M
    """
    import random
    words = list(wv_from_bin.vocab.keys())
    print("Shuffling words ...")
    random.seed(224)
    random.shuffle(words)
    words = words[:10000]
    print("Putting %i words into word2ind and matrix M..." % len(words))
    word2ind = {}
    M = []
    curInd = 0
    for w in words:
        try:
            M.append(wv_from_bin.word_vec(w))
            word2ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    for w in required_words:
        if w in words:
            continue
        try:
            M.append(wv_from_bin.word_vec(w))
            word2ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    M = np.stack(M)
    print("Done.")
    return M, word2ind


def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurence count matrix of dimensionality (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the following SVD function from Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html

        Params:
            M (numpy matrix of shape (number of unique words in the corpus , number of unique words in the corpus)): co-occurence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of corpus words, k)): matrix of k-dimensioal word embeddings.
                    In terms of the SVD from math class, this actually returns U * S
    """
    n_iters = 10  # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))

    svd = TruncatedSVD(n_components=k, n_iter=n_iters)
    M_reduced = svd.fit_transform(M)

    print("Done.")
    return M_reduced


def plot_embeddings(M_reduced, word2ind, words, model_name):
    """ Plot in a scatterplot the embeddings of the words specified in the list "words".
        NOTE: do not plot all the words listed in M_reduced / word2ind.
        Include a label next to each point.

        Params:
            M_reduced (numpy matrix of shape (number of unique words in the corpus , 2)): matrix of 2-dimensioal word embeddings
            word2ind (dict): dictionary that maps word to indices for matrix M
            words (list of strings): words whose embeddings we want to visualize
    """

    for i, word in enumerate(words):
        index = word2ind[word]
        embedding = M_reduced[index]

        x, y = embedding[0], embedding[1]
        plt.scatter(x, y, marker='x', color='red')
        plt.text(x, y, word, fontsize=9)
    plt.title('Word Vector Visualization for {} embeddings'.format(model_name))
    # plt.show()
    plt.savefig(os.path.join(OUTPUT_DIR, 'WordVectorVisualizations-{}.png'.format(model_name)), bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    models = ["word2vec-google-news-300", 'glove-twitter-200']

    for model in models:
        print('\nModel = {}\n'.format(model))

        vector_embeddings = load_embedding_model(model)
        M, word2ind = get_matrix_of_vectors(vector_embeddings)

        n_components = 2
        M_reduced = reduce_to_k_dim(M, k=n_components)
        assert M_reduced.shape[1] == n_components, "Expected # of columns in reduced vector metrics to be {} but was {}".format(n_components, M_reduced.shape[1])

        plot_embeddings(M_reduced, word2ind, ['portland', 'pacific', 'forest', 'ocean', 'city', 'food', 'green', 'weird', 'cool', 'oregon'], model)

        # Words with Multiple Meanings
        for word in ['arms', 'once', 'subject', 'key']:
            print('Words that are most similar to word "{}" = {}'.format(word, vector_embeddings.most_similar(word)))

        # Synonyms & Antonyms
        for word_set in [('happy', 'excited', 'sad'), ('long', 'lengthy', 'short')]:
            w1, w2, w3 = word_set
            distance_between_antonyms = vector_embeddings.distance(w1, w3)
            distance_between_synonyms = vector_embeddings.distance(w1, w2)
            if distance_between_antonyms < distance_between_synonyms:
                print('Antonyms ({}, {}) are closer [distance = {}] than synonyms ({}, {}) [distance = {}]'.format(w1, w3, distance_between_antonyms, w1, w2,
                                                                                                                   distance_between_synonyms))

        # analogies
        if 'twitter' not in model:
            for word_set in [('Italy', 'Rome', 'Germany'), ('Recession', 'Poor', 'Boom')]:
                a1, b1, a2 = word_set
                print('Analogies {}:{}, {}:\n{}'.format(a1, b1, a2, (vector_embeddings.most_similar(positive=[a2, b1], negative=[a1]))))

        # biases
        for word_set in [('woman', 'doctor', 'man'), ('man', 'doctor', 'woman'),
                         ('caucasian', 'police officer', 'black'), ('black', 'police officer', 'caucasian'),
                         ('woman', 'computer programmer', 'man'), ('man', 'computer programmer', 'woman'),
                         ('woman', 'resume', 'man'), ('man', 'resume', 'woman')]:
            a1, b1, a2 = word_set
            print('Analogies {}:{}, {}:\n{}'.format(a1, b1, a2, (vector_embeddings.most_similar(positive=[a2, b1], negative=[a1]))))

print('test')
