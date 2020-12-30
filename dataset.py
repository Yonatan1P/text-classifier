import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

filepath_dict = { 'yelp': 'data/sentiment_analysis/yelp_labelled.txt',
'amazon': 'data/sentiment_analysis/amazon_cells_labelled.txt',
'imdb': 'data/sentiment_analysis/imdb_labelled.txt'}

df_list = []
for source, filepath in filepath_dict.item():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='/t')
    df['source'] = source
    df_list.append(df)

df = pd.concat(df_list)

sentences = ['John likes ice cream', 'John hates chocolate.']

vectorizer = CountVectorizer(min_df=0, lowercase=False)
vectorizer.fit(sentences)
vectorizer.vocabulary_
vectorizer.transform(sentences).toarray()
