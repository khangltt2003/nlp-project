import re
import nltk
import string
import requests
import pandas as pd
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('universal_tagset')
# nltk.download('averaged_perceptron_tagger_eng')

lemmatizer = WordNetLemmatizer()
def load_special_words_and_reviews():
    stopwords_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
    return set(stopwords_list.decode().splitlines())

#remove stop words
stop_words = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", 
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", 
    "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", 
    "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", 
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", 
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", 
    "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", 
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", 
    "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", 
    "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", 
    "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", 
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "d", "ll", "s", "ve", "t", "re",
    "aren", "can", "counldn", "didn", "doesn", "don", "hadn", "hasn", "haven", "m", "isn", "let", "mustn", "shan", "shouldn",
    "wasn", "weren", "won", "wouldn",
}

# def get_wordnet_pos(tag):
#     if tag.startswith('J'):  
#         return 'a'
#     elif tag.startswith('V'):  
#         return 'v'
#     elif tag.startswith('N'):  
#         return 'n'
#     elif tag.startswith('R'): 
#         return 'r'
#     else:
#         return None 

def preprocessData():
    print("preprocessing data")
    review_df = pd.read_pickle("../../dataset/reviews_segment.pkl")[["review_id", "review_text"]]
    ids = []
    preprocessed_reviews = []
    count = 1
    for review in review_df.itertuples():
        if count % 10000 == 0:
            print(f"processed {count} reviews")
        count+=1
        ids.append(review.review_id.replace("'", ""))
        preprocessed_reviews.append(" ".join(preprocess(review.review_text)))
    
    cleaned_df  = pd.DataFrame({"review_id" : ids, "review_text": preprocessed_reviews})
    cleaned_df.to_pickle("./../preprocessed_data/processed_data.pkl")

def preprocess(text: str):
    #lower text
    text = text.lower()

    #remove html tags
    html_tags = r'<[^>]+>'
    text = re.sub(html_tags, '', text)
    
    #remove urls
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub(r'', text)
    
    #remove emojis
    emoji_pattern = r'^(?:[\u2700-\u27bf]|(?:\ud83c[\udde6-\uddff]){1,2}|(?:\ud83d[\udc00-\ude4f]){1,2}|[\ud800-\udbff][\udc00-\udfff]|[\u0021-\u002f\u003a-\u0040\u005b-\u0060\u007b-\u007e]|\u3299|\u3297|\u303d|\u3030|\u24c2|\ud83c[\udd70-\udd71]|\ud83c[\udd7e-\udd7f]|\ud83c\udd8e|\ud83c[\udd91-\udd9a]|\ud83c[\udde6-\uddff]|\ud83c[\ude01-\ude02]|\ud83c\ude1a|\ud83c\ude2f|\ud83c[\ude32-\ude3a]|\ud83c[\ude50-\ude51]|\u203c|\u2049|\u25aa|\u25ab|\u25b6|\u25c0|\u25fb|\u25fc|\u25fd|\u25fe|\u2600|\u2601|\u260e|\u2611|[^\u0000-\u007F])+$'
    text = re.sub(emoji_pattern, '', text)
    
    #remove punctuations
    table = str.maketrans({char: ' ' for char in string.punctuation})
    text = text.translate(table)

    #remove stop words    
    text_tokens = text.split()
    text_tokens = [word for word in text_tokens if word not in stop_words]

    # lemmatize text
    lemmatized_tokens = [lemmatizer.lemmatize(token , pos = "v") for token in text_tokens]
    
    # pos_tags = pos_tag(text_tokens, tagset= "universal")
    # lemmatized_tokens = []
    # for word, tag in pos_tags:
    #     wordnet_pos = get_wordnet_pos(tag) or 'n'
    #     lemmatized_tokens.append(lemmatizer.lemmatize(word, pos=wordnet_pos))

    return lemmatized_tokens

# text =  "walk, she'll he will Walking. https://www.youtube.com/ i you we they walked/ walks? The quick brown foxes jumped over the lazy dogs. Walking slowly, they enjoyed the view."
# # text = 'It is not down on any map; true places never are.'
# print(preprocess(text))
