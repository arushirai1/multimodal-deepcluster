
import pickle
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
import string
import heapq
import re

def get_dataset(df):
    dataset = []
    d2 = {}
    for key, item in df.items():
        d2[key] = ' '.join(list(item['text']))
    for key, item in d2.items():
        item = clean_doc(item)
        item = preprocess(item)
        dataset.append(item)

    dataset = [' '.join(data) for data in dataset] # lists of list to one list
    return dataset

def preprocess(item):
    for i in range(len(item)):
        item[i] = item[i].lower()
        item[i] = re.sub(r'\W', ' ', item[i])
        item[i] = re.sub(r'\s+', ' ', item[i])
    return item

def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

with open('coin_howto_overlap_captions.pickle','rb') as f:
    captions_df = pickle.load(f)
    word2count = {}
    dataset = get_dataset(captions_df)

    # count word occurrence
    for data in dataset:
        words = nltk.word_tokenize(data)
        for word in words:
            if word not in word2count.keys():
                word2count[word] = 1
            else:
                word2count[word] += 1
    # get top 300 freq words
    freq_words = heapq.nlargest(300, word2count, key=word2count.get)

with open('freq_words.pickle', 'wb') as f:
    pickle.dump(freq_words, f)

