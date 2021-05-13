# Import Necessary Libraries
import re
import nltk
import numpy as np
import itertools
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords

class CustomerReviewKeywords:
    # Method to preprocess the data
    stopwords_ = list(set(stopwords.words("english")))
    stopwords_+=['voodoo','doughnuts','doughnut','voodoodoughnut']
    def preprocess(self,data,stopwords):
        # Preprocessing Texts
        preprocessed_texts = []
        lem = WordNetLemmatizer()
        # Cleaing the data, removing stopwords
        for sent in data:
            sent = sent.replace('\\r', ' ')
            sent = sent.replace('\\"', ' ')
            sent = sent.replace('\\n', ' ')
            sent = re.sub('[^A-Za-z ]+', ' ', sent)
            # lemmatizing
            sent=' '.join(lem.lemmatize(word) for word in sent.split() if word not in stopwords)
            preprocessed_texts.append(sent.lower().strip())
        return preprocessed_texts
    
    # https://towardsdatascience.com/keyword-extraction-with-bert-724efca412ea
    def max_sum_sim(self,doc_embedding, candidate_embeddings, candidates, top_n, nr_candidates):
        # Calculate distances and extract keywords
        distances = cosine_similarity(doc_embedding, candidate_embeddings)
        distances_candidates = cosine_similarity(candidate_embeddings, 
                                            candidate_embeddings)

        # Get top_n words as candidates based on cosine similarity
        words_idx = list(distances.argsort()[0][-nr_candidates:])
        words_vals = [candidates[index] for index in words_idx]
        distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

        # Calculate the combination of words that are the least similar to each other
        min_sim = np.inf
        candidate = None
        for combination in itertools.combinations(range(len(words_idx)), top_n):
            sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
            if sim < min_sim:
                candidate = combination
                min_sim = sim
        return [words_vals[idx] for idx in candidate]

    # Diversify the keywords using max sum similarity, higher the value of nr_candidates higher the diversity
    def extract_keywords_bert_diverse(self,doc,stopwords,top_n=5,nr_candidates=10):
        n_gram_range = (1,1)
        # Extract candidate words/phrases using count vectorizer (TF-IDF Scores)
        count = CountVectorizer(ngram_range=n_gram_range, stop_words=stopwords).fit([doc])
        candidates = count.get_feature_names()
        # Embeddings of the document using Bert    
        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        doc_embedding = model.encode([doc])
        candidate_embeddings = model.encode(candidates)
        keywords=self.max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n, nr_candidates)
        return keywords 

    # Method to get the trending keywords
    def get_trending_keywords(self,data_most_reviewed_store,num_keywords=10):
        # Stopwards
        stopwords_ = list(set(stopwords.words("english")))
        stopwords_+=['voodoo','doughnuts','doughnut','voodoodoughnut']
        # Filtering the dataset based on Review Sentiments
        positive_reviews=data_most_reviewed_store[data_most_reviewed_store['sentiments']==1]
        negative_reviews=data_most_reviewed_store[data_most_reviewed_store['sentiments']==3]
        neutral_reviews=data_most_reviewed_store[data_most_reviewed_store['sentiments']==2]
        preprocessed_texts_neg=self.preprocess(negative_reviews.text.values,stopwords_)
        preprocessed_texts_pos=self.preprocess(positive_reviews.text.values,stopwords_)
        preprocessed_texts_neu=self.preprocess(neutral_reviews.text.values,stopwords_)  
        keywords={}
        corpus=' '.join(preprocessed_texts_pos[-550::])
        keywords['positive']=self.extract_keywords_bert_diverse(corpus,stopwords_,num_keywords)
        corpus=' '.join(preprocessed_texts_neg[-550::])
        keywords['negative']=self.extract_keywords_bert_diverse(corpus,stopwords_,num_keywords)
        return keywords   


