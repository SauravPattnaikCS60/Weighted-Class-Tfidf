import re
import os
import sys
import warnings
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from wcbtfidf import Wcbtfidf

warnings.filterwarnings("ignore")


def set_session_path():
    """
        Set session path to current working directory

    """
    module_path = os.path.abspath(os.path.join("../.."))
    if module_path not in sys.path:
        sys.path.append(module_path)


def read_and_prepare_data(filename):
    """
        Sample the dataset in an imbalancd way
        Compare imbalance statistics

    """
    df = pd.read_csv(filename)

    print(f"Shape of the original dataset is {df.shape}")
    print(
        f"Target distribution of the original dataset \n{df.sentiment.value_counts(normalize=True)}"
    )

    df["sentiment"] = df["sentiment"].map({"negative": 0, "positive": 1})

    # Sampling data with imbalance
    # Total sample: 25k points
    # 23k points -> class 0
    # 2k points -> class 1

    negative_samples = df[df["sentiment"] == 0].sample(n=23000, random_state=60)
    positive_samples = df[df["sentiment"] == 1].sample(n=2000, random_state=60)

    # Merge and shuffle the imbalanced data
    final_df = pd.concat([negative_samples, positive_samples]).sample(
        frac=1, random_state=60
    )

    print(f"Final data shape is {final_df.shape}")
    print(
        f"Final target distribution is \n{final_df.sentiment.value_counts(normalize=True)}"
    )
    return final_df


def preprocess_text(text):
    """
        Sanitize input

    """
    text = text.lower()
    text = re.sub("[^a-z0-9]", " ", text)
    text = re.sub("(\s)+", " ", text)
    return text


def get_model(xtrain, xtest, ytrain, ytest, max_feat, model, type_model):
    """
        Run model on imbalanced dataset

    """
    if type_model == "TFIDF":
        print("Running TFIDF")
        tfidf = TfidfVectorizer(max_features=max_feat, stop_words="english")
        train_df = pd.DataFrame(
            tfidf.fit_transform(xtrain).toarray(), columns=tfidf.vocabulary_
        )
        test_df = pd.DataFrame(
            tfidf.transform(xtest).toarray(), columns=tfidf.vocabulary_
        )

        model.fit(train_df, ytrain)
        preds_tfidf = model.predict(test_df)
        print(classification_report(ytest, preds_tfidf))

        return tfidf
    
    if type_model == "WCBTFIDF":
        print("Running WCBTFIDF")
        wcbtfidf = Wcbtfidf(max_features=max_feat)
        wcbtfidf.fit(xtrain, ytrain)

        train_df = wcbtfidf.transform(xtrain)
        test_df = wcbtfidf.transform(xtest)

        model.fit(train_df, ytrain)
        preds_wcbtfidf = model.predict(test_df)
        print(classification_report(ytest, preds_wcbtfidf))

        return wcbtfidf


def check_hypothesis(xtrain, xtest, ytrain, ytest, max_feat, model):
    """
        Evaluate and return model objects

    """

    tfidf = get_model(xtrain, xtest, ytrain, ytest, max_feat, model, "TFIDF")

    wcbtfidf = get_model(xtrain, xtest, ytrain, ytest, max_feat, model, "WCBTFIDF")

    return wcbtfidf, tfidf


def analysis(wcbtfidf_object, tfidf_object):
    """
        Vocab comparision statistics

    """

    # Length Comparison

    tfidf_vocab = tfidf_object.vocabulary_
    wcbtfidf_vocab = wcbtfidf_object.combine_vocab

    print(len(wcbtfidf_vocab), len(tfidf_vocab))

    # Words that are present in tfidf vocab but not in wcbtfidf

    print(list(set(tfidf_vocab) - set(wcbtfidf_vocab)))

    # Words that are present in wcbtfidf but not in tfidf

    print(list(set(wcbtfidf_vocab) - set(tfidf_vocab)))


def main():

    set_session_path()

    final_df = read_and_prepare_data("imdb_dataset.csv")
    final_df["clean_text"] = final_df["review"].apply(preprocess_text)

    final_df = final_df[["clean_text", "sentiment"]]
    xtrain, xtest, ytrain, ytest = train_test_split(
        final_df["clean_text"],
        final_df["sentiment"],
        test_size=0.25,
        random_state=60,
        stratify=final_df["sentiment"],
    )

    print(xtrain.shape, ytrain.shape)
    print(xtest.shape, ytest.shape)

    model = LogisticRegression()
    wcbtfidf_object, tfidf_object = check_hypothesis(
        xtrain, xtest, ytrain, ytest, 300, model
    )

    analysis(wcbtfidf_object, tfidf_object)


if __name__ == "__main__":
    main()
