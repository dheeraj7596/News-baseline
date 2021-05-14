import pickle


def populate_candidate_hashtags(tweet_candidate_hashtags):
    if i not in tweet_candidate_hashtags:
        tweet_candidate_hashtags[i] = {}
        tweet_candidate_hashtags[i]["positive"] = []
        tweet_candidate_hashtags[i]["negative"] = []
    else:
        if "positive" not in tweet_candidate_hashtags[i]:
            tweet_candidate_hashtags[i]["positive"] = []

        if "negative" not in tweet_candidate_hashtags[i]:
            tweet_candidate_hashtags[i]["negative"] = []
    return tweet_candidate_hashtags


if __name__ == "__main__":
    # data_path = "./data/"
    data_path = "/data/dheeraj/News-baseline/"
    train_df = pickle.load(open(data_path + "train_df.pkl", "rb"))
    test_df = pickle.load(open(data_path + "test_df.pkl", "rb"))

    tweet_candidate_hashtags = pickle.load(open(data_path + "tweet_candidate_hashtags.pkl", "rb"))
    news_candidate_hashtags = pickle.load(open(data_path + "news_candidate_hashtags.pkl", "rb"))
    domain_candidate_hashtags = pickle.load(open(data_path + "domain_candidate_hashtags.pkl", "rb"))
    random_walk_train_hashtags = pickle.load(open(data_path + "random_walk_train_hashtags.pkl", "rb"))

    train_hashtags = {}
    for i in range(len(train_df)):
        tweet_candidate_hashtags = populate_candidate_hashtags(tweet_candidate_hashtags)
        news_candidate_hashtags = populate_candidate_hashtags(news_candidate_hashtags)
        domain_candidate_hashtags = populate_candidate_hashtags(domain_candidate_hashtags)
        random_walk_train_hashtags = populate_candidate_hashtags(random_walk_train_hashtags)

        train_hashtags[i] = {}
        train_hashtags[i]["positive"] = list(set(tweet_candidate_hashtags[i]["positive"] + news_candidate_hashtags[i][
            "positive"] + domain_candidate_hashtags[i]["positive"] + random_walk_train_hashtags[i]["positive"]))
        train_hashtags[i]["negative"] = list(set(tweet_candidate_hashtags[i]["negative"] + news_candidate_hashtags[i][
            "negative"] + domain_candidate_hashtags[i]["negative"] + random_walk_train_hashtags[i]["negative"]))

    pickle.dump(train_hashtags, open(data_path + "train_hashtags.pkl", "wb"))
