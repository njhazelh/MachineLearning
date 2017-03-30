"""
This script will create and run a HMM (Hidden Markov Model).
"""

from collections import defaultdict
import numpy as np
import math
import hw3.hmm.data_reader as dr

RARE_WORD = '_RARE_'

class HiddenMarkovModel:
    """
    This class represents a Hidden Markov Model that can be used
    to tag sequences of data (eg. tag words in a sentence with
    their part of speech).
    
    This algorithm minimizes the sum of the log probabilities
    rather than the product of the probabilities.  Doing so
    prevents the possibility of underflow issues for long
    sentences.
    """

    def __init__(self, tags):
        """
        Init the model
        :param tags: A list containing the legal tags (excluding START and STOP).
        """
        self.tags = tags

    def fit(self, tag_probs, grams):
        """
        Setup the training data in the model. Unlink other algorithms,
        this doesn't do much.
        :param tag_probs: A dictionary mapping words -> tag -> prob(word|tag)
        :param grams: A dictionary mapping tag trigrams to probabilities.
            depths must be ordered such that
            grams[tag_k2][tag_k1][tag_k] = p(tag_k|tag_k-1=tag_k1, tag_k-2=tag_k2)
        :return: `self` to make chaining easy
        """
        self.tag_probs = tag_probs
        self.grams = grams
        return self

    def calc_prob(self, probs, k, tag_k2, tag_k1, tag_k, word):
        """
        Get the probability of seeing tag_k at point k, given the
        word being tagged, tag_k2, tag_k1, and previous probabilities.
        :param probs: A dict containing the probailities of lower levels of the
            viterbi algorithm.
            probs[k][u][v] = max(calc_prob(k-1,w,u) * q(v|w,u) * e(x_k, v)) for all w
        :param k: The index of the word
        :param tag_k2: The tag two indexes prior
        :param tag_k1: The tag one index prior
        :param tag_k: The tag of the word
        :param word: The word
        :return: The sum of the log probabilities of seeing this data.
        """
        if word not in self.tag_probs:
            word = RARE_WORD
        try:
            prev = probs[k-1][tag_k2][tag_k1]
            q = self.grams[tag_k2][tag_k1][tag_k]
            e = self.tag_probs[word][tag_k]
            return prev + e + q
        except KeyError:
            return -math.inf

    def word(self, k, words):
        """
        Get the word at index k, with appropriate tranformations
        :param k: The index to get
        :param words: The list of words
        :return: The word, lowercased and RARE-ified.
        """
        word = words[k].lower()
        return RARE_WORD if word not in self.tag_probs else word

    def predict(self, words):
        """
        Find the best sequence of characters to represent this data.
        :param words: A list of words
        :return: A list of tags
        """
        if len(words) == 0:
            return []

        num_words = len(words)
        probs = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: -math.inf
        )))
        ptrs = defaultdict(lambda: defaultdict(dict))
        probs[0][dr.WordTag.START][dr.WordTag.START] = 0

        for k in range(1, num_words+1):
            word = self.word(k - 1, words)
            # We don't have to look at all possible tags, because some have 0 probability given the word.
            k2_tags = [dr.WordTag.START] if k < 3 else self.tag_probs[self.word(k - 3, words)].keys()
            k1_tags = [dr.WordTag.START] if k == 1 else self.tag_probs[self.word(k - 2, words)].keys()
            k_tags = self.tag_probs[self.word(k - 1, words)].keys()
            for tag_k1 in k1_tags:
                for tag_k in k_tags:
                    best = max((
                        (self.calc_prob(probs, k, tag_k2, tag_k1, tag_k, word), tag_k2)
                        for tag_k2 in k2_tags), key=lambda x: x[0])
                    probs[k][tag_k1][tag_k] = best[0]
                    ptrs[k][tag_k1][tag_k] = best[1]

        if len(words) == 1:
            best = None, None
            for tag in self.tags:
                try:
                    prob = probs[num_words][dr.WordTag.START][tag] \
                           + self.grams[dr.WordTag.START][tag][dr.WordTag.STOP]
                except KeyError:
                    prob = -math.inf
                if best[0] is None or prob > best[0]:
                    best = prob, tag
            return [best[1]]


        # Find the tags for the last two characters
        best_pair = None
        for tag_k2 in ptrs[num_words].keys():
            for tag_k1 in ptrs[num_words][tag_k2].keys():
                try:
                    prob = probs[num_words][tag_k2][tag_k1] \
                           + self.grams[tag_k2][tag_k1][dr.WordTag.STOP]
                except KeyError:
                    prob = -math.inf
                if best_pair is None or prob > best_pair[0]:
                    best_pair = prob, tag_k2, tag_k1
        labels = [best_pair[2], best_pair[1]]

        # Backtrace the tags for each word
        for k in range(num_words-2, 0, -1):
            labels.append(ptrs[k+2][labels[-1]][labels[-2]])
        labels.reverse()

        return labels

def clean_data(tags, grams):
    """
    Do some data manipulation before training
    :param tags: A dict containing counts of words and tags.
        tags[word][tag] = count(word with tag)
    :param grams: A dict containing the counts of different N-Grams
        tags[3][w][u][v] = count(WUV)
        tags[2][u][v] = count(UV)
        tags[1][v] = count(V)
    :return:
        tags where
            - The words have been lowercased
            - Words with less than 5 occurences are filed under RARE
            - counts are converted to log(p(word|tag))
        grams where
            - counts are converted to log(p(w|UV))
    """
    remove_words = []
    for word in {word for word in tags if word != word.lower()}:
        word_lower = word.lower()
        if word_lower not in tags:
            tags[word_lower] = {}
        for tag in tags[word]:
            if tag not in tags[word_lower]:
                tags[word_lower][tag] = tags[word][tag]
            else:
                tags[word_lower][tag] += tags[word][tag]
        remove_words.append(word)

    # Remove words which were lowercased
    for word in remove_words:
        del tags[word]

    # Find rare words and move them to the rare-words
    tags[RARE_WORD] = {}
    remove_words = []
    for word in tags:
        if sum(tags[word].values()) < 5:
            for tag in tags[word]:
                if tag not in tags[RARE_WORD]:
                    tags[RARE_WORD][tag] = tags[word][tag]
                else:
                    tags[RARE_WORD][tag] += tags[word][tag]
            remove_words.append(word)

    # Remove the less common words
    for word in remove_words:
        del tags[word]

    # Turn count(word with tag) into e(word|tag)
    for word in tags:
        for tag in tags[word]:
            word_tag_count = tags[word][tag]
            tag_count = grams[1][tag]
            tags[word][tag] = math.log(word_tag_count / tag_count)

    # Turn count(WUV) into q(V|WU)
    for tag_k2 in grams[3].keys():
        for tag_k1 in grams[3][tag_k2].keys():
            for tag_k in grams[3][tag_k2][tag_k1].keys():
                grams[3][tag_k2][tag_k1][tag_k] = \
                    math.log(grams[3][tag_k2][tag_k1][tag_k] / grams[2][tag_k2][tag_k1])

    return tags, grams

def main():
    """
    Run the HMM on the collection of words and sentence parts.
    """
    # Load the training data
    tags, grams = dr.read_training_data("data/UD_English/train.counts")
    # Open the solution file, so we can check the accuracy as we go.
    _, test_soln = dr.read_test_soln("data/UD_English/test.tags")
    # Open the test features file
    sentences = dr.read_sentences("data/UD_English/test.words")
    tags, grams = clean_data(tags, grams)

    model = HiddenMarkovModel([
        dr.WordTag.ADJ,
        dr.WordTag.ADV,
        dr.WordTag.INTJ,
        dr.WordTag.NOUN,
        dr.WordTag.PROPN,
        dr.WordTag.VERB,
        dr.WordTag.ADP,
        dr.WordTag.AUX,
        dr.WordTag.CONJ,
        dr.WordTag.DET,
        dr.WordTag.NUM,
        dr.WordTag.PART,
        dr.WordTag.PRON,
        dr.WordTag.SCONJ,
        dr.WordTag.PUNCT,
        dr.WordTag.SYM,
        dr.WordTag.X
    ]).fit(tags, grams[3])

    total = 0.0
    accurate = 0.0
    solns = []
    for i, sentence in enumerate(sentences):
        if i % 20 == 0 and total > 0:
            print("progress: %d/%d, accuracy: %d/%d = %.4f%%" \
                  % (i, len(sentences), accurate, total, accurate/total * 100))
        pred = model.predict(sentence)
        total += len(pred)
        if len(pred) == 0 and pred[0] == test_soln[i][0]:
            accurate += 1
        else:
            matches = np.array(pred) == np.array(test_soln[i])
            accurate += matches.sum()
        solns.append(pred)
    dr.write_output(sentences, solns, 'output/test.soln')

if __name__ == "__main__":
    main()
