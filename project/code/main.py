import pandas as pd
import argparse

from inverted_index import Inverted_Index

negative_words = []
positive_words = []
review_df = []
def main():
    parser = argparse.ArgumentParser(description="Perform the boolean search.")
    
    parser.add_argument("-a1", "--aspect1", type=str, required=True, default=None, help="First word of the aspect")
    parser.add_argument("-a2", "--aspect2", type=str, required=True, default=None, help="Second word of the aspect")
    parser.add_argument("-o", "--opinion", type=str, required=True, default=None, help="Only word of the opinion")
    parser.add_argument("-m", "--method", type=str, required=True, default=None, help="The method of boolean operation. Methods\
                                            can be method1, method2 or method3")
    # Parse the arguments
    args = parser.parse_args()
    a1 = args.aspect1
    a2 = args.aspect2
    o = args.opinion
    
    inverted_index  = Inverted_Index()

    inverted_index.load()

    if args.method.lower() == "method1":
        inverted_index.method1(a1, a2, o)
        
    elif args.method.lower() == "method2":
        inverted_index.method2(a1, a2, o)
        
    elif args.method.lower() == "method3":
        inverted_index.method3(a1, a2, o)
    else:
        print("\n!! The method is not supported !!\n")
        return	
      
if __name__ == "__main__":
    main()

