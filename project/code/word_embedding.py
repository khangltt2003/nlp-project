import os
import pandas as pd
import numpy as np
from numpy.linalg import norm
from gensim.models import Word2Vec
from preprocess import preprocess

class WordEmbedding:
    def __init__(self):
        self.dataset_path = "./../../dataset/processed_data.pkl"
        self.model_path = "./../trained_models/w2v_model.model"
        self.embedding_path = "./../trained_models/review_embeddings.npy"
        self.review_ids_path = "./../trained_models/review_ids.npy"
        self.model = None
        self.review_ids = None
        self.embedded_reviews = None

    def setup(self):
        if os.path.exists(self.model_path):
            self.load_model_and_data()
            return
          
        reviews = pd.read_pickle(self.dataset_path).values
        review_ids = []
        tokenized_reviews = []

        for review in reviews:
            review_ids.append(review[0])
            tokenized_reviews.append(review[1].split())

        print("training word embedding model...")
        w2v = Word2Vec(sentences=tokenized_reviews, vector_size=100, window=10, workers=4, min_count=1, seed=1)

        print("generating review embedding vector for each review...")
        review_embeddings = [self.get_review_embedding(review, w2v) for review in tokenized_reviews]

        w2v.save(self.model_path)
        np.save(self.review_ids_path, np.array(review_ids))
        np.save(self.embedding_path, np.array(review_embeddings))

    def load_model_and_data(self):
        self.model = Word2Vec.load(self.model_path)
        self.review_ids = np.load(self.review_ids_path)
        self.embedded_reviews = np.load(self.embedding_path)

    def cosine_similarity(self, a, b):
        norm_a = norm(a)
        norm_b = norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0
        return np.dot(a, b) / (norm_a * norm_b)

    def get_review_embedding(self, review, model):
        all_words_in_review_embedding_vector = [
            model.wv[word] for word in review if word in model.wv
        ]
        review_embedding_vector = (
            np.mean(all_words_in_review_embedding_vector, axis=0)
            if all_words_in_review_embedding_vector
            else np.zeros(model.vector_size)
        )
        return review_embedding_vector

    def search(self, query):
        self.load_model_and_data()
        tokenized_query = preprocess(query)
        embedded_query = self.get_review_embedding(tokenized_query, self.model)
        similarities = [self.cosine_similarity(embedded_query, embedded_review) for embedded_review in self.embedded_reviews]

        id_with_similarity = list(zip(self.review_ids, similarities))
        sorted_similarity = sorted(id_with_similarity, key=lambda x: x[1], reverse=True)
        best_matched_reviews = [review for review in sorted_similarity if review[1] > 0.7]
        print(pd.DataFrame(best_matched_reviews, columns=["review_id", "similarity"]))
        return best_matched_reviews

# wordEmbed = WordEmbedding()

# results = wordEmbed.search("audio quality poor")

# print(pd.DataFrame(results, columns=["review_id", "similarity"]))
