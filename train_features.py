import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import Counter
from scipy import sparse
from fast_pagerank import pagerank


def create_features():
    pass


def train_vectorizer(train_df):
    corpus = list(train_df["clean_tweet"]) + list(train_df["news"].dropna())
    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)
    pickle.dump(vectorizer, open(data_path + "tfidf_vectorizer.pkl", "wb"))


def create_train_test(df):
    train_df, test_df = train_test_split(df, test_size=0.1)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    pickle.dump(train_df, open(data_path + "train_df.pkl", "wb"))
    pickle.dump(test_df, open(data_path + "test_df.pkl", "wb"))


def tweet_candidate_hashtags(train_df):
    vectorizer = pickle.load(open(data_path + "tfidf_vectorizer.pkl", "rb"))
    train_tweet_vec = vectorizer.transform(train_df["clean_tweet"])
    tweet_cos_sims = train_tweet_vec.dot(train_tweet_vec.T).todense()
    n = tweet_cos_sims.shape[0]
    train_hashtags = {}
    for i in range(n):
        print(i, "out of ", n)
        inds = np.array(np.argsort(tweet_cos_sims[i]))[0][::-1][:50]
        temp = []
        for j in inds:
            temp += train_df["hashtag"][j].split(";")
        temp_hashtags = set(dict(sorted(dict(Counter(temp)).items(), key=lambda item: -item[1])[:20]).keys())
        positive_hashtags = train_df["hashtag"][i].split(";")

        train_hashtags[i] = {}
        train_hashtags[i]["positive"] = list(set(positive_hashtags).intersection(temp_hashtags))
        train_hashtags[i]["negative"] = list(temp_hashtags - set(positive_hashtags))
    pickle.dump(train_hashtags, open(data_path + "tweet_candidate_hashtags.pkl", "wb"))


def news_candidate_hashtags(train_df):
    vectorizer = pickle.load(open(data_path + "tfidf_vectorizer.pkl", "rb"))
    news_ind_to_original_ind = {}
    count = 0
    news_list = []
    for i, row in train_df.iterrows():
        if not isinstance(row["news"], str):
            continue
        else:
            news_list.append(row["news"])
            news_ind_to_original_ind[count] = i
            count += 1

    train_news_vec = vectorizer.transform(news_list)
    news_cos_sims = train_news_vec.dot(train_news_vec.T).todense()
    n = news_cos_sims.shape[0]
    train_hashtags = {}
    for i in range(n):
        print(i, "out of ", n)
        inds = np.array(np.argsort(news_cos_sims[i]))[0][::-1][:50]
        temp = []
        for j in inds:
            temp += train_df["hashtag"][news_ind_to_original_ind[j]].split(";")
        temp_hashtags = set(dict(sorted(dict(Counter(temp)).items(), key=lambda item: -item[1])[:20]).keys())
        positive_hashtags = train_df["hashtag"][i].split(";")

        train_hashtags[news_ind_to_original_ind[i]] = {}
        train_hashtags[news_ind_to_original_ind[i]]["positive"] = list(
            set(positive_hashtags).intersection(temp_hashtags))
        train_hashtags[news_ind_to_original_ind[i]]["negative"] = list(temp_hashtags - set(positive_hashtags))

    for i in range(len(train_df)):
        if i in train_hashtags:
            continue
        else:
            train_hashtags[i] = {}
            train_hashtags[i]["positive"] = []
            train_hashtags[i]["negative"] = []

    pickle.dump(train_hashtags, open(data_path + "news_candidate_hashtags.pkl", "wb"))


def create_domain_dic(train_df):
    temp_df = train_df[train_df['domain'].notna()].reset_index(drop=True)
    domain_dic = {}
    for i, row in temp_df.iterrows():
        try:
            domain_dic[row["domain"]] += row["hashtag"].split(";")
        except:
            domain_dic[row["domain"]] = row["hashtag"].split(";")

    for i in domain_dic:
        domain_dic[i] = list(dict(sorted(dict(Counter(domain_dic[i])).items(), key=lambda item: -item[1])[:20]).keys())

    pickle.dump(domain_dic, open(data_path + "domain_dic.pkl", "wb"))


def candidate_hashtags_domain(train_df):
    domain_dic = pickle.load(open(data_path + "domain_dic.pkl", "rb"))
    train_hashtags = {}
    for i, row in train_df.iterrows():
        train_hashtags[i] = {}
        if not isinstance(row["domain"], str):
            train_hashtags[i]["positive"] = []
            train_hashtags[i]["negative"] = []
        else:
            train_hashtags[i]["positive"] = list(
                set(train_df["hashtag"][i].split(";")).intersection(set(domain_dic[row["domain"]])))
            train_hashtags[i]["negative"] = list(set(domain_dic[row["domain"]]) - set(train_hashtags[i]["positive"]))
    pickle.dump(train_hashtags, open(data_path + "domain_candidate_hashtags.pkl", "wb"))


def create_entity_hashtag_graph(train_df):
    ent_id = {}
    id_ent = {}
    ent_words = set([])
    for i, row in train_df.iterrows():
        if isinstance(row["news_entity"], list):
            ent_words.update(set(row["news_entity"]))
        if isinstance(row["tweet_entity"], list):
            ent_words.update(set(row["tweet_entity"]))

    ent_words = list(ent_words)
    for i, w in enumerate(ent_words):
        ent_id[w] = i
        id_ent[i] = w

    start = len(ent_words)

    hash_id = {}
    id_hash = {}
    hash_words = set([])
    for i, row in train_df.iterrows():
        hash_words.update(set(row["hashtag"].split(";")))

    hash_words = list(hash_words)
    for i, word in enumerate(hash_words):
        hash_id[word] = start + i
        id_hash[start + i] = word

    edges = []
    weights = []
    for i, row in train_df.iterrows():
        entities = []
        if isinstance(row["news_entity"], list):
            entities += list(set(row["news_entity"]))
        if isinstance(row["tweet_entity"], list):
            entities += list(set(row["tweet_entity"]))

        hashtags = row["hashtag"].split(";")

        for e in entities:
            for h in hashtags:
                edges.append([ent_id[e], hash_id[h]])
                weights.append(1)

    edges = np.array(edges)
    node_count = len(ent_id) + len(hash_id)
    G = sparse.csr_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(node_count, node_count))
    pickle.dump(ent_id, open(data_path + "ent_id.pkl", "wb"))
    pickle.dump(id_ent, open(data_path + "id_ent.pkl", "wb"))
    pickle.dump(hash_id, open(data_path + "hash_id.pkl", "wb"))
    pickle.dump(id_hash, open(data_path + "id_hash.pkl", "wb"))
    pickle.dump(G, open(data_path + "graph.pkl", "wb"))


def get_random_walk_candidate_hashtags(train_df):
    ent_id = pickle.load(open(data_path + "ent_id.pkl", "rb"))
    id_hash = pickle.load(open(data_path + "id_hash.pkl", "rb"))
    G = pickle.load(open(data_path + "graph.pkl", "rb"))

    start = len(ent_id)
    count = len(ent_id) + len(id_hash)
    train_hashtags = {}
    for ind, row in train_df.iterrows():
        print(ind)
        entities = []
        if isinstance(row["news_entity"], list):
            entities += list(set(row["news_entity"]))
        if isinstance(row["tweet_entity"], list):
            entities += list(set(row["tweet_entity"]))
        personalized = np.zeros((count,))
        for e in entities:
            personalized[ent_id[e]] = 1
        pr = pagerank(G, p=0.85, personalize=personalized)
        temp_list = list(pr)[start:]
        args = np.argsort(temp_list)[::-1][:20]
        top_hashtags = []
        for i in args:
            top_hashtags.append(id_hash[start + i])

        positive_hashtags = row["hashtag"].split(";")

        train_hashtags[ind] = {}
        train_hashtags[ind]["positive"] = list(set(positive_hashtags).intersection(set(top_hashtags)))
        train_hashtags[ind]["negative"] = list(set(top_hashtags) - set(positive_hashtags))

    pickle.dump(train_hashtags, open(data_path + "random_walk_train_hashtags.pkl", "wb"))


if __name__ == "__main__":
    base_path = "./data/"
    # base_path = "/data/dheeraj/News-baseline/"
    dataset = "2020"
    data_path = base_path + dataset + "/"
    # df = pickle.load(open(data_path + "tweet_news_with_domain_2020.pkl", "rb"))
    # create_train_test(df)

    train_df = pickle.load(open(data_path + "train_df.pkl", "rb"))
    test_df = pickle.load(open(data_path + "test_df.pkl", "rb"))

    # create_domain_dic(train_df)
    # train_vectorizer(train_df)

    tweet_candidate_hashtags(train_df)
    news_candidate_hashtags(train_df)
    candidate_hashtags_domain(train_df)

    create_entity_hashtag_graph(train_df)
    get_random_walk_candidate_hashtags(train_df)
    pass
