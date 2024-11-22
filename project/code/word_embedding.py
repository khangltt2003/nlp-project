import os
import pandas as pd
import numpy as np
from numpy.linalg import norm
from gensim.models import Word2Vec
from utils.preprocess import preprocess

def cosine_similarity(a,b ):
    return np.dot(a,b)/(norm(a)*norm(b))

def get_review_embedding(review, model):
    all_words_in_review_embedding_vector = [model.wv[word] for word in review]
    review_embedding_vector = np.mean(all_words_in_review_embedding_vector, axis=0) if all_words_in_review_embedding_vector else np.zeros(model.vector_size)
    return review_embedding_vector

def setup():
    reviews = pd.read_pickle("../../dataset/reviews_segment.pkl")[["review_id", "review_text"]].values
    
    review_ids = []
    tokenized_reviews = []
    for review in reviews:
        review_ids.append(review[0].replace("'", ""))
        tokenized_reviews.append(preprocess(review[1]))
    
    print("Training word embedding model...")
    w2v = Word2Vec(sentences=tokenized_reviews, vector_size=100, window=10, workers=4, min_count=1, seed=1)

    print("Generating review embedding vector for each review...")
    review_embeddings = []
    for review in tokenized_reviews:
        review_embeddings.append(get_review_embedding(review, w2v))

    w2v.save("../../trained_models/w2v_model.model")
    np.save("../../trained_models/review_ids.npy", np.array(review_ids))
    np.save("../../trained_models/review_embeddings.npy", np.array(review_embeddings))
        
def find_relevant_reviews(query):
    if not os.path.exists("../../trained_models/w2v_model.model"):
        setup()
        
    review_ids = np.load("../../trained_models/review_ids.npy")
    embedded_reviews = np.load("../../trained_models/review_embeddings.npy")
    model = Word2Vec.load("../../trained_models/w2v_model.model")
    
    tokenized_query = preprocess(query)
    embedded_query = get_review_embedding(tokenized_query, model)
    similarities = [cosine_similarity(embedded_query, embedded_review) for embedded_review in embedded_reviews]
    
    id_with_similarity = list(zip(review_ids, similarities))
    
    sorted_similarity = sorted(id_with_similarity, key=lambda x: x[1], reverse=True)

    best_matched_reviews = [review[0] for review in  sorted_similarity]
    
    return best_matched_reviews
    
setup()

res =  find_relevant_reviews("audio quality poor")
print(pd.DataFrame(res))
