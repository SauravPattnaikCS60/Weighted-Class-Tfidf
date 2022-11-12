import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class Wcbtfidf:

    def __init__(self, max_features,custom_weights=None):
        self.max_features = max_features
        self.combine_vocab = []
        self.class_wise_vocab = {}
        self.final_tfidf = None
        self.label_dict = custom_weights

    def fit(self, X, y):

        if not isinstance(y,pd.Series):
            y = pd.Series(y)

        if not isinstance(X,pd.Series):
            X = pd.Series(X)

        if self.label_dict == None:
            self.label_dict = y.value_counts(normalize=True).to_dict()
            for key, val in self.label_dict.items():
                new_val = int(np.round(val* self.max_features,1))
                self.label_dict[key] = new_val

        elif len(self.label_dict.keys()) != y.nunique():
            raise ValueError("Custom weights keys and number of unique labels should match")

        elif np.sum(list(self.label_dict.values())) != self.max_features:
            raise ValueError("Sum of custom weights and max features do not match")

        else:
            pass


        self.combine_vocab = self.return_total_vocab(X, y, self.label_dict)
        self.final_tfidf = TfidfVectorizer(vocabulary=self.combine_vocab, stop_words='english')
        self.final_tfidf.fit(X)

    def transform(self, X):
        transformed_data = self.final_tfidf.transform(X)
        transformed_data = pd.DataFrame(transformed_data.toarray(), columns=self.final_tfidf.get_feature_names_out())
        return transformed_data

    def return_total_vocab(self, X, y, label_dict):

        exclude = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves",
                   "you", "your", "yours", "yourself", "yourselves", "he", "him",
                   "his", "himself", "she", "her", "hers", "herself", "it", "its",
                   "itself", "they", "them", "their", "theirs", "themselves", "what",
                   "which", "who", "whom", "this", "that", "these", "those", "am", "is",
                   "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
                   "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because",
                   "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between",
                   "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
                   "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there",
                   "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
                   "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
                   "will",
                   "just", "don", "should", "now"]

        total_vocab = []
        for key, val in label_dict.items():
            if val != 0:
                slice_data = X[y == key]
                tfidf = TfidfVectorizer(max_features=val, stop_words=exclude)
                tfidf.fit(slice_data)
                vocab = list(tfidf.get_feature_names_out())
                total_vocab.extend(vocab)
                exclude.extend(vocab)
                self.class_wise_vocab[key] = vocab
            else:
                self.class_wise_vocab[key] = []

        return total_vocab
