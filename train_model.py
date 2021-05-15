import pickle
import itertools
import numpy as np
from sklearn.svm import SVC

if __name__ == "__main__":
    # data_path = "./data/"
    data_path = "/data/dheeraj/News-baseline/"

    train_df = pickle.load(open(data_path + "train_df.pkl", "rb"))
    test_df = pickle.load(open(data_path + "test_df.pkl", "rb"))

    tweet_candidate_hashtags = pickle.load(open(data_path + "tweet_candidate_hashtags.pkl", "rb"))
    news_candidate_hashtags = pickle.load(open(data_path + "news_candidate_hashtags.pkl", "rb"))
    domain_candidate_hashtags = pickle.load(open(data_path + "domain_candidate_hashtags.pkl", "rb"))
    random_walk_train_hashtags = pickle.load(open(data_path + "random_walk_train_hashtags.pkl", "rb"))

    train_X = []
    train_y = []

    for i in range(len(train_df)):
        print(i)
        feature_vecs = {}
        # feature-vec = [tweet, news, domain, random]

        hashtags = set(train_df["hashtag"][i].split(";"))
        for h in hashtags:
            temp = []
            if h in tweet_candidate_hashtags[i]["positive"]:
                temp.append(1)
            else:
                temp.append(0)

            if h in news_candidate_hashtags[i]["positive"]:
                temp.append(1)
            else:
                temp.append(0)

            if h in domain_candidate_hashtags[i]["positive"]:
                temp.append(1)
            else:
                temp.append(0)

            if h in random_walk_train_hashtags[i]["positive"]:
                temp.append(1)
            else:
                temp.append(0)

            feature_vecs[h] = np.array(temp)

        neg_entities = list(set(tweet_candidate_hashtags[i]["negative"] + news_candidate_hashtags[i]["negative"] +
                                random_walk_train_hashtags[i]["negative"]))
        for h in neg_entities:
            temp = []
            if h in tweet_candidate_hashtags[i]["positive"]:
                temp.append(1)
            else:
                temp.append(0)

            if h in news_candidate_hashtags[i]["positive"]:
                temp.append(1)
            else:
                temp.append(0)

            if h in domain_candidate_hashtags[i]["positive"]:
                temp.append(1)
            else:
                temp.append(0)

            if h in random_walk_train_hashtags[i]["positive"]:
                temp.append(1)
            else:
                temp.append(0)

            feature_vecs[h] = np.array(temp)

        positive_samples = itertools.product(list(hashtags), neg_entities)
        neg_samples = itertools.product(neg_entities, list(hashtags))

        for p in positive_samples:
            train_X.append(feature_vecs[p[0]] - feature_vecs[p[1]])
            train_y.append(1)

        for p in neg_samples:
            train_X.append(feature_vecs[p[0]] - feature_vecs[p[1]])
            train_y.append(0)

    pickle.dump(train_X, open(data_path + "train_X.pkl", "wb"))
    pickle.dump(train_y, open(data_path + "train_y.pkl", "wb"))

    clf = SVC()
    clf.fit(train_X, train_y)
    pickle.dump(clf, open(data_path + "clf.pkl", "wb"))
