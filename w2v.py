import pandas as pd
import re
import string
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk import word_tokenize
import warnings
warnings.filterwarnings('ignore')
from gensim.models import Word2Vec, FastText
# import sklearn
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix
from stop_words import get_stop_words
from nltk.corpus import stopwords
# from scipy import spatial
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader
from sklearn.metrics import classification_report,accuracy_score

# from gensim.corpora import Dictionary
# from gensim.models.tfidfmodel import TfidfModel
# from sklearn.decomposition import PCA
# from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
# from gensim.similarities.annoy import AnnoyIndexer
def clean_tweets(df):
    
    punctuations = string.punctuation
    
    df.loc[:, 'tweet'] = df.tweet.str.replace('@USER', '') #Remove mentions (@USER)
    df.loc[:, 'tweet'] = df.tweet.str.replace('URL', '') #Remove URLs
    df.loc[:, 'tweet'] = df.tweet.str.replace('&amp', 'and') #Replace ampersand (&) with and
    df.loc[:, 'tweet'] = df.tweet.str.replace('&lt','') #Remove &lt
    df.loc[:, 'tweet'] = df.tweet.str.replace('&gt','') #Remove &gt
    df.loc[:, 'tweet'] = df.tweet.str.replace('\d+','') #Remove numbers
    df.loc[:, 'tweet'] = df.tweet.str.lower() #Lowercase

    #Remove punctuations
    for punctuation in punctuations:
        df.loc[:, 'tweet'] = df.tweet.str.replace(punctuation, " ")

    #Remove emojis
    df.loc[:, 'tweet'] = df.astype(str).apply(
        lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii')
    )
    df.loc[:, 'tweet'] = df.tweet.str.strip() 
    stop_words = list(stopwords.words('english'))
    pat = r'\b(?:{})\b'.format('|'.join(stop_words))
    df['tweet'] = df['tweet'].str.replace(pat, " ")
    df['tweet'] = df['tweet'].str.replace(r'\s+', " ")
    

def compare_text_word2vec(file_one,file_two,all_data,k): #not 1 off 2
    
    off_tweet = []
    not_tweet = []
    #Getting the topk most occuring words in file_one(i.e., NOT_SUBSET)
    for line in file_one:
        for word in line:
            not_tweet.append(word)
    not_tweet = Counter(not_tweet)
    topk_not =[k for k,v in not_tweet.most_common(k)]

    #Getting the topk most occuring words in file_two(i.e., OFF_SUBSET)
    for line in file_two:
        for word in line:
            off_tweet.append(word)
    off_tweet = Counter(off_tweet)
    topk_off =[k for k,v in off_tweet.most_common(k)] #return the k most common word

    data = []
    for sent in all_data:
        new_sent = []
        for word in sent:
            new_word = word.lower()
            if new_word[0] not in string.punctuation:
                new_sent.append(new_word)
        if len(new_sent) > 0:
            data.append(new_sent)
    #Load pretrained model in this case glove-twitter-25
    w2v1 = gensim.downloader.load('glove-twitter-25')

    #Training word2vec model for next part of assignment and saving it for further use.
    w2v1_all = Word2Vec(sentences = data,vector_size = 10,window = 50,epochs = 200)
    print("OFF:",topk_off)
    print("NOT:",topk_not)
    total = 0

    c=0
    for i, word in enumerate(topk_not):
        for j,words in enumerate(topk_off):
            total += w2v1.similarity(word,topk_off[j])
            print(word,"similarity:",topk_off[j],"is",w2v1.similarity(word,topk_off[j]))
            c+=1

    w2v1.save("word2vec_pretrained.model")
    w2v1_all.save("word2vec_new.wordvectors")
    print("word2vec similarity score for",k,":",total/c)
    w = []
    s = set(topk_not+topk_off)
    for i in s:
        w += w2v1.most_similar(i,topn=10)
        print("\nMost similar to",i,"are:\n",w2v1.most_similar(i,topn=10),"\n")


    #NOTE:approach changed. 
    # max1 = Counter(" ".join(file_one).split()).most_common(k) #Get most common words from file 1
    # max2 = Counter(" ".join(file_two).split()).most_common(k) #Get most common words from file 2
    # max1 = Counter(file_one)
    # max2 = Counter(file_two)
    # print(type(max1))
    # print(max2)
    # dict1 = dict(max1)
    # dict2 = dict(max2)
    # print(dict1)
    # dict1a = list(dict1.keys())
    # dict2a = list(dict2.keys())
    # all_data = dict1a+dict2a
    # print(all_data)
    # print(w2v1)
    # print(w2v1.wv.similarity(topk_off,topk_not))
        # print(word)
    # w2v1.train(data,total_examples=len(data),epochs = 10)
    # w2v1.build_vocab(all_data)
    # tokens = list(w2v1.wv.index_to_key)
    # print(tokens)
    # csine = cosine_similarity(w2v1.wv.vectors)
    # print(csine)
    # output = create_dataframe(csine,tokens)
    # print(output)
    # Output = create_dataframe(csine,['doc_1','doc_2'])
    # print(Output)
    # print(cosine_similarity(w2v1.wv.vectors))
    # # print(w2v1.wv.vectors)
    # w2v2 = Word2Vec()
    # w2v2.build_vocab(dict2.keys())
    # w2v2.build_vocab(dict2.keys())
    # print(w2v2.wv.vectors)


    #NOTE:Approach changed. This was done to use nltk to clean the words and train the word2vec and find similarity 
    #based on the similarity of words not the occurance. I had interpreted the question differently. 
    
    # clean_corpus=[]
    # stop_words = list(get_stop_words('english'))         #About 900 stopwords
    # nltk_words = list(stopwords.words('english')) #About 150 stopwords
    
    # for w in corpus1:
    #     for x in w:
    #         if(x in stop_words):
    #             clean_corpus.append(w)
    # print(corpus1)
    # corpus2=[]
    # for col in file_two.tweet:
    #     word_list = col.split(" ")
    #     corpus2.append(word_list)
    # clean_corpus2=[]
    # stop_words = list(get_stop_words('english'))         #About 900 stopwords
    # nltk_words = list(stopwords.words('english')) #About 150 stopwords
    # for w in corpus2:
    #     for x in w:
    #         if(x not in stop_words):
    #             clean_corpus2.append(w)
    # # print(clean_corpus2)
    # model1 = Word2Vec(vector_size=200, window=10, min_count=5, workers=11, alpha=0.025, epochs=20)
    # model1.build_vocab(clean_corpus)
    # model1.train(clean_corpus, total_examples=model1.corpus_count, epochs=model1.epochs)
    # model1.save('C:/Users/bssup/Documents/fall22/NLP/hw3/models/corpus_word2vec1.model')
    # # model1 = Word2Vec(corpus_file=clean_corpus, vector_size=20, min_count=1)
    # dictionary = Dictionary(clean_corpus)
    # tfidf = TfidfModel(dictionary=dictionary)
    # words = [word for word, count in dictionary.most_common()]
    # word_vectors = model1.wv.vectors_for_all(words, allow_inference=False)

    # indexer = AnnoyIndexer(word_vectors, num_trees=2)  # use Annoy for faster word similarity lookups
    # termsim_index = WordEmbeddingSimilarityIndex(word_vectors, kwargs={'indexer': indexer})
    # similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary, tfidf)  #
    # print(similarity_matrix)

    # model2 = Word2Vec(vector_size=200, window=10, min_count=5, workers=11, alpha=0.025, epochs=20)
    # model2.build_vocab(clean_corpus2)
    # model2.train(clean_corpus2, total_examples=model2.corpus_count, epochs=model2.epochs)
    # model2.save('C:/Users/bssup/Documents/fall22/NLP/hw3/models/corpus_word2vec2.model')

    # similarwords_file1 = model1.wv.index_to_key[:k]
    # similarwords_file2 = model2.wv.index_to_key[:k]
    # print(similarwords_file1)
    # print(similarwords_file2)
    # print(sklearn.metrics.pairwise.cosine_similarity(similarwords_file1, Y=similarwords_file2))
    # termsim_index = model.wmdistance(model2,model)
    # print(termsim_index)
    
def create_subset(file):
    tweet = pd.read_csv(file,sep='\t')
    tweet = tweet.iloc[1:,:]
    clean_tweets(tweet)
    #creating subset of data based on their type
    off = tweet[tweet['subtask_a'] == 'OFF']
    nott = tweet[tweet['subtask_a'] == 'NOT']
    data_OFF = [list(map(str.lower, word_tokenize(sent))) for sent in off['tweet']]
    data_NOT = [list(map(str.lower, word_tokenize(sent))) for sent in nott['tweet']]
    all_data = data_NOT + data_OFF

    return data_OFF,data_NOT,all_data
    
if __name__ == '__main__':
    filename = "C:/Users/bssup/Documents/fall22/NLP/hw2/archive/olid-training-v1.0.tsv"
    OFF_subset,NOT_subset,all_data=create_subset(filename)
    compare_text_word2vec(NOT_subset,OFF_subset,all_data,k=5)
    compare_text_word2vec(NOT_subset,OFF_subset,all_data,k=10)
    compare_text_word2vec(NOT_subset,OFF_subset,all_data,k=20)