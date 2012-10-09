#!/usr/bin/env python3

from twentiment.thirdparty.probability import FreqDist
from twentiment.naivebayes import NaiveBayesClassifier
from twentiment.text import normalize_text


def learn_main():
    pos_tweets = [('I love this car', 'positive'),
                  ('This view is amazing', 'positive'),
                  ('I feel great this morning', 'positive'),
                  ('I am so excited about the concert', 'positive'),
                  ('He is my best friend', 'positive')]

    neg_tweets = [('I do not like this car', 'negative'),
                  ('This view is horrible', 'negative'),
                  ('I feel tired this morning', 'negative'),
                  ('I am not looking forward to the concert', 'negative'),
                  ('He is my enemy', 'negative')]

    tweets = []
    for (words, sentiment) in pos_tweets + neg_tweets:
        tweets.append((normalize_text(words), sentiment))

    features = list(get_word_features(get_words(tweets)))
    training_set = [(extract_features(doc, features), label) for (doc, label)
                    in tweets]

    classifier = NaiveBayesClassifier.train(training_set)

    while True:
        line = input("twentiment > ")
        if not line:
            break

        tweet = normalize_text(line)
        twfeat = extract_features(tweet, features)

        prob_result = classifier.prob_classify(twfeat)
        score = prob_result.prob('positive') - prob_result.prob('negative')

        print("Sentiment: {} ({}%)".format(prob_result.max(), score * 100))


def get_words(tweets):
    all_words = []
    for (words, _) in tweets:
        all_words.extend(words)

    return all_words


def get_word_features(wordlist):
    return FreqDist(wordlist).keys()


def extract_features(document, all_features):
    return {word: (word in set(document)) for word in all_features}


if __name__ == "__main__":
    learn_main()
