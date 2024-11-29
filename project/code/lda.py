import os
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
from preprocess import preprocess


class TopicBasedSearch:
    def __init__(self):
        self.dataset_path = "./../preprocessed_data/processed_data.pkl"
        self.topics_file_path = "../../dataset/Topics.txt"
        self.lda_model_path =  "./../trained_models/lda_model.gensim"
        self.topics = {}
        self.reviews = []
        self.lda_model = None
        self.lda_topics = {}

    def load_data(self):
        review_df = pd.read_pickle(self.dataset_path)
        self.reviews = review_df["review_text"].values

    # def generate_topics(self):
    #     with open(self.topics_file_path, "r") as f:
    #         lines = f.readlines()
    #         for line in lines:
    #             topic = line[:line.find(".")]
    #             words = line[line.find(".") + 2:]
    #             self.topics[topic] = set(preprocess(words))

    def map_review_to_topic(self, preprocessed_review):
        topic_scores = {
            topic_name: len(set(preprocessed_review) & set(topic_terms))
            for topic_name, topic_terms in self.topics.items()
        }
        best_topic = max(topic_scores, key=topic_scores.get)
        return best_topic, topic_scores[best_topic]

    def train_lda(self):
        print("loading data...")
        self.load_data()
        print("training lda model for topic based search...")
        tokenized_reviews = [preprocess(review) for review in self.reviews]
        dictionary = corpora.Dictionary(tokenized_reviews)
        corpus = [dictionary.doc2bow(review) for review in tokenized_reviews]
        
        self.lda_model = LdaModel(corpus=corpus, num_topics= 100, id2word=dictionary,random_state=1)
        self.lda_topics = {
            f"Topic {i + 1}": [word for word, _ in self.lda_model.show_topic(i, topn=100)]
            for i in range(100)
        }
        print(self.lda_topics)
        self.lda_model.save(self.lda_model_path)
        print("training done!!!")
        
    def setup(self):
        if os.path.exists(self.lda_model_path):
            self.lda_model = LdaModel.load(self.lda_model_path)
        else:
            self.train_lda()

    def search(self, query):
        print("search relevant reviews for query", query)
        preprocessed_query = preprocess(query)
        best_topic, _ = self.map_review_to_topic(preprocessed_query)
        relevant_reviews = [
            review
            for review in self.reviews
            if self.map_review_to_topic(preprocess(review))[0] == best_topic
        ]
        return relevant_reviews


# review_search = TopicBasedSearch()
# review_search.load_data()
# review_search.generate_topics()
# review_search.load_lda_model()

# query = "I want a Blu-ray player with excellent video quality."
# relevant_reviews = review_search.search(query)

# print("Query:", query)
# print("Relevant Reviews:", relevant_reviews)
