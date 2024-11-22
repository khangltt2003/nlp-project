import pandas as pd
import json
# from utils import preprocessor

# review_text = []

# review_df = pd.read_pickle("../../dataset/reviews_segment.pkl")[["review_text"]]

# print(review_text)

# all_text = set()
# for review in review_df.itertuples():
#     all_text.add(preprocessor.preprocess(review.review_text))

# print(all_text)


posting_list = pd.read_pickle("../output/posting_lists.pkl")
posting_list_dict = json.loads(posting_list)

words = sorted(list(set(posting_list_dict.keys())))

print(len(words))

# print(len(posting_list_dict.keys()))



    