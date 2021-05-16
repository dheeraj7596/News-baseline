import pickle
import numpy as np
import itertools

if __name__ == "__main__":
    # base_path = "./data/"
    base_path = "/data/dheeraj/News-baseline/"
    dataset = "2020"
    data_path = base_path + dataset + "/"

    test_df = pickle.load(open(data_path + "test_df.pkl", "rb"))

    tweet_candidate_hashtags = pickle.load(open(data_path + "test_tweet_candidate_hashtags.pkl", "rb"))
    news_candidate_hashtags = pickle.load(open(data_path + "test_news_candidate_hashtags.pkl", "rb"))
    domain_candidate_hashtags = pickle.load(open(data_path + "test_domain_candidate_hashtags.pkl", "rb"))
    random_walk_train_hashtags = pickle.load(open(data_path + "test_random_walk_train_hashtags.pkl", "rb"))

    clf = pickle.load(open(data_path + "clf_0.1.pkl", "rb"))

    # - get features of each candidate
    # - all pairs
    # - pass it through classifier
    # - calculate max and pick top-5

    f = open("./out_2018.txt", "w")
    for i in range(len(test_df)):
        test_X = []
        print(i)
        feature_vecs = {}
        # feature-vec = [tweet, news, domain, random]

        candidates = list(set(tweet_candidate_hashtags[i] + news_candidate_hashtags[i] + domain_candidate_hashtags[i] + \
                              random_walk_train_hashtags[i]))

        for h in candidates:
            temp = []
            if h in tweet_candidate_hashtags[i]:
                temp.append(1)
            else:
                temp.append(0)

            if h in news_candidate_hashtags[i]:
                temp.append(1)
            else:
                temp.append(0)

            if h in domain_candidate_hashtags[i]:
                temp.append(1)
            else:
                temp.append(0)

            if h in random_walk_train_hashtags[i]:
                temp.append(1)
            else:
                temp.append(0)

            feature_vecs[h] = np.array(temp)

        pairs = list(itertools.combinations(candidates, 2))
        for p in pairs:
            test_X.append(feature_vecs[p[0]] - feature_vecs[p[1]])

        preds = clf.predict(test_X)

        reco_scores = {}
        for pred_index, pred in enumerate(preds):
            if pred == 1:
                try:
                    reco_scores[pairs[pred_index][0]] += 1
                except:
                    reco_scores[pairs[pred_index][0]] = 1

        top_hashtags = list(dict(sorted(reco_scores.items(), key=lambda item: -item[1])[:10]).keys())
        f.write(";".join(top_hashtags))
        f.write("\n")

    f.close()
