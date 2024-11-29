import os 
import pandas as pd
import argparse
from inverted_index import BooleanSearch
from lda import TopicBasedSearch
from preprocess import preprocessData
from word_embedding import WordEmbedding
# from word_embedding import find_relevant_reviews


topic_model =  TopicBasedSearch()
boolean_search = BooleanSearch()
word_embedding = WordEmbedding()

def setup():
    if not os.path.exists("./../preprocessed_data/processed_data.pkl"):
        print("preprocessing data...")
        preprocessData() 

    boolean_search.setup()
    word_embedding.setup()
    topic_model.setup()
    
def main():
    setup()
    query = input("Enter query (a1 a2 o): ")
    while(query != "exit"):
        print("1. Boolean Search using \"a1 AND a2 AND o\"")
        print("2. Word Embedding using Word2Vec")
        print("3. Topic Model using LDA")
        method  = int(input("Choose 1 method: "))
        if method == 1:
            boolean_search.search(query)
        elif method == 2:
            word_embedding.search(query)
        elif method == 3:
            topic_model.search(query)
        else:
            print("invalid method")
            
        query = input("Enter query (a1 a2 o): ")
  
    # parser = argparse.ArgumentParser(description="Perform the boolean search.")
    # parser.add_argument("-a1", "--aspect1", type=str, required=True, default=None, help="First word of the aspect")
    # parser.add_argument("-a2", "--aspect2", type=str, required=True, default=None, help="Second word of the aspect")
    # parser.add_argument("-o", "--opinion", type=str, required=True, default=None, help="Only word of the opinion")
    # parser.add_argument("-m", "--method", type=str, required=True, default=None, help="The method of boolean operation. Methods\
    #                                         can be method1, method2 or method3")
    # # Parse the arguments
    # args = parser.parse_args()
    # a1 = args.aspect1
    # a2 = args.aspect2
    # o = args.opinion
    
    # boolean_search  = BooleanSearch()

    # boolean_search.load()

    # if args.method.lower() == "method1":
    #     boolean_search.method1(a1, a2, o)
        
    # elif args.method.lower() == "method2":
    #     boolean_search.method2(a1, a2, o)
        
    # elif args.method.lower() == "method3":
    #     boolean_search.method3(a1, a2, o)
    # else:
    #     print("\n!! The method is not supported !!\n")
    #     return	  
    
      
if __name__ == "__main__":
    main()

