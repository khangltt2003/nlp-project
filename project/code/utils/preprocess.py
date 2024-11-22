import re
import nltk
import string
import requests
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# nltk.download('stopwords')
# nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def load_special_words_and_reviews():
    stopwords_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
    return set(stopwords_list.decode().splitlines())

# remove stop words
stop_words = load_special_words_and_reviews()

def preprocess(text: str):
    #lower text
    text = text.lower()

    #remove html tags
    html_tags = r'<[^>]+>'
    text = re.sub(html_tags, '', text)
    
    # remove urls
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub(r'', text)
    
    # remove punctuations
    table = str.maketrans({char: ' ' for char in string.punctuation})
    text = text.translate(table)
    
    #remove emojis
    emoji_pattern = r'^(?:[\u2700-\u27bf]|(?:\ud83c[\udde6-\uddff]){1,2}|(?:\ud83d[\udc00-\ude4f]){1,2}|[\ud800-\udbff][\udc00-\udfff]|[\u0021-\u002f\u003a-\u0040\u005b-\u0060\u007b-\u007e]|\u3299|\u3297|\u303d|\u3030|\u24c2|\ud83c[\udd70-\udd71]|\ud83c[\udd7e-\udd7f]|\ud83c\udd8e|\ud83c[\udd91-\udd9a]|\ud83c[\udde6-\uddff]|\ud83c[\ude01-\ude02]|\ud83c\ude1a|\ud83c\ude2f|\ud83c[\ude32-\ude3a]|\ud83c[\ude50-\ude51]|\u203c|\u2049|\u25aa|\u25ab|\u25b6|\u25c0|\u25fb|\u25fc|\u25fd|\u25fe|\u2600|\u2601|\u260e|\u2611|[^\u0000-\u007F])+$'
    text = re.sub(emoji_pattern, '', text)
    
    word_tokens = text.split()
    text_tokens = [word for word in word_tokens if word not in stop_words]
    
    # lemmatize text
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word, pos ='v') for word in text_tokens]

    return lemmatized_tokens

# text =  "walk, she'll he will Walking. https://chatgpt.com/c/6731316d-6624-800c-86c8-3cbafb6f680c i you we they walked/ walks?"
# # text = 'It is not down on any map; true places never are.'
# print(preprocess(text))

