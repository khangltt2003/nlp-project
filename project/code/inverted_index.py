import json
import pickle
import os
import string
import requests
import pandas as pd
from utils.preprocess import preprocess

class Inverted_Index:
    def __init__(self):
        self.inverted_index = {} 
        self.map = {}
        
    def generate_inverted_index(self):
        print("loading reviews")
        review_df = pd.read_pickle("./../../dataset/reviews_segment.pkl")[["review_id", "review_text"]]
      
        print("generating inverted index...")
        # table = str.maketrans({char: ' ' for char in string.punctuation if char != "'"})
        # print(table)
        # stopwords = load_special_words_and_reviews()
        count = 1
        for review in review_df.itertuples():
            if count % 10000 == 0:
                print(f"processed {count} reviews...") 
            count += 1
            
            #preprocess review text
            cleaned_review = preprocess(review.review_text)
            
            review_id = review.review_id.replace("'", "") 
            
            self.map[review_id] = review.review_text
            
            for key in cleaned_review:
                #key exist ?  get posting list : initialize empty set to key
                #add review_id to key's posting list
                self.inverted_index.setdefault(key, set()).add(review_id)

        print("done")
        self.save()
    
    #for words that not in inverted_index
    def add_to_inverted_index(self, words):
        review_df = pd.read_pickle("./../../dataset/reviews_segment.pkl")[["review_id", "review_text"]]
      
        for review in review_df.itertuples():
            cleaned_text = preprocess(review.review_text)
            
            for key in cleaned_text:
                if key not in words: continue
                
                review_id = review.review_id.replace("'", "")
                self.inverted_index.setdefault(key, set()).add(review_id)
        self.save()
    
    #check if word in inverted index
    def check_and_generate(self, words):
        unvisited_word = []
        for word in words:
            if word not in self.inverted_index:
                unvisited_word.append(word)
                
        if not unvisited_word: return
        
        #generate posting lists for words that not in inverted index
        self.add_to_inverted_index(unvisited_word)
        
    def AND_operation(self, l1, l2):
        answer = []
        l1 = list(l1)
        l2 = list(l2)
        l1.sort()
        l2.sort()
        i = 0
        j = 0
        while i < len(l1) and j < len(l2):
            if(l1[i] == l2[j]):
                answer.append(l1[i])
                i+=1
                j+=1
            elif(l1[i] < l2[j]):
                i+=1
            else:
                j+= 1
        
        return answer

    def OR_operation(self, l1, l2):
        l1 = list(l1)
        l2 = list(l2)
        return list(set(l1+l2))
    
    #a1 or a2 or o
    def method1(self, a1, a2, o):
        print(f"perform boolean search for query: '{a1}' or '{a2}' or '{o}'")
        self.check_and_generate([a1, a2, o])
        res =  self.OR_operation(self.OR_operation(self.inverted_index[a1], self.inverted_index[a2]), self.inverted_index[o])
        self.save_print_query_res(a1, a2, o, "method1", res)
        res = [{"review_id" : id, "review_text" : self.map[id]} for id in res]
        return res

    #a1 and a2 and o
    def method2(self, a1, a2, o):
        print(f"perform boolean search for query: '{a1}' and '{a2}' and '{o}'")
        self.check_and_generate([a1, a2, o])
        res =  self.AND_operation(self.AND_operation(self.inverted_index[a1], self.inverted_index[a2]), self.inverted_index[o])
        self.save_print_query_res(a1, a2, o, "method2", res)
        res = [{"review_id" :id, "review_text" : self.map[id]} for id in res]
        return res
        
        
    #a1 or a2 and o
    def method3(self, a1, a2, o):
        print(f"perform boolean search for query: '{a1}' or '{a2}' and '{o}'")
        self.check_and_generate([a1, a2, o])
        res =  self.AND_operation(self.OR_operation(self.inverted_index[a1], self.inverted_index[a2]), self.inverted_index[o])
        self.save_print_query_res(a1, a2, o, "method3", res)
        res = [{"review_id" :id, "review_text" : self.map[id]} for id in res]
        return res
        
        
    def load(self):
        if(os.path.exists("./../output/posting_lists.pkl")):
            # get posting list
            with open('./../output/posting_lists.pkl', 'rb') as file:
                json_data = pickle.load(file)
                self.inverted_index = json.loads(json_data)
            
            #get review id and review text
            review_df = pd.read_pickle("./../../dataset/reviews_segment.pkl")[["review_id", "review_text"]]
            for review in review_df.itertuples():
                review_id = review.review_id.replace("'", "")
                self.map[review_id] = review.review_text
            
        else:
            self.generate_inverted_index()
            
    def save_print_query_res(self,a1, a2, o, m, res):
        revs = pd.DataFrame()
        revs["review_index"] = [r for r in res]
        print(revs)
        # path = f"../output/" + a1 + "_" + a2 + "_" + o + "_" + m + ".pkl"
        # print(f"saving result to {path}")
        # revs.to_pickle(path)
        print("done")
    
    def save(self):
        print("save inverted index to ../output/posting_lists.pkl")
        serializer  = {key: list(value) for key, value in self.inverted_index.items()}
        json_data = json.dumps(serializer)
        with open('./../output/posting_lists.pkl', 'wb') as file:
            pickle.dump(json_data, file)
        print("Done")