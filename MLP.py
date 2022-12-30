from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk
import pandas as pd
import re
import numpy as np
from gensim.models import Word2Vec
from nltk.lm.preprocessing import padded_everygram_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report,accuracy_score
import gensim
import warnings
warnings.filterwarnings('ignore')

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

 
#Function is trasform the data into vectors using word2vec
def create_vector(S):
        vect_s = []
        for doct in S:
            tokenized_vect_s = []
            for sent in nltk.sent_tokenize(doct):
                tokenized_vect_s += nltk.word_tokenize(sent)
            vect_s.append(np.array(tokenized_vect_s))
        return np.array(vect_s)

#Function is used to create embedding. 
def MEV(transformed, model):
    dimension = 10
    transformed_t = create_vector(transformed)
    return np.array([
            np.mean([model.wv[w] for w in words if w in model.wv]
                    or [np.zeros(dimension)], axis=0)
            for words in transformed_t
        ])
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

    return all_data
#EXTRA CREDIT PART
def update_embeddings(file_path):
    all_data = create_subset(file_path)
    
    data = []
    for sent in all_data:
        new_sent = []
        for word in sent:
            new_word = word.lower()
            if new_word[0] not in string.punctuation:
                new_sent.append(new_word)
        if len(new_sent) > 0:
            data.append(new_sent)
    w2v1_all = Word2Vec(sentences = data,vector_size = 10,window = 50,epochs = 200)
    return w2v1_all
    

def train_MLP(train_path,k):
    tweet = pd.read_csv(train_path,sep='\t')
    tweet_data = tweet[['tweet']] #Extract tweets
    tweets= tweet[['subtask_a']] #Extract subtsak_a labels
    tweets.columns.values[0] = 'type' #Rename class attribute
    tweet_data = tweet_data.join(tweets)
    clean_tweets(tweet_data)
    w2v = update_embeddings(train_path) # Model with embeddings from the training file is used here.[Extra credit part]
    # w2v = Word2Vec.load("./word2vec_new.model") #load word2vec from the saved vector[alternative: is to use the saved model from w2v file.]
    X = MEV(tweet_data['tweet'], w2v)
    Y = tweet_data['type']
    Y.columns = 'type'
    model = MLPClassifier(k)
    model.fit(X,Y)
    print(model)
    return model,w2v #Model and w2v is returned to get the test_MLP function

def Test_MLP(path,model,w2v,k):
    tweet =  pd.read_csv(path,sep='\t')
    tweet = tweet.iloc[1:,:]
    clean_tweets(tweet)
    levela = pd.read_csv('C:/Users/bssup/Documents/fall22/NLP/hw2/archive/labels-levela.csv')
    levela = levela.iloc[:,-1]
    X = MEV(tweet['tweet'], w2v)
    y_pred = model.predict(X)
    result_df = pd.DataFrame(columns = ['tweet','probability','Classification','True_class'])
    result_df['tweet'] = tweet['tweet']
    result_df['probability'] = model.predict_proba(X)
    result_df['Classification'] = y_pred
    result_df['True_class'] = levela
    # print(result_df)
    print(classification_report(levela,y_pred))
    result_df.to_csv('./MLP'+str(k)+'_OUTPUT_1.csv',index=False)
    #TEST_MLP creates 3 csv files with labes(MLP1_output,MLP2_output,MLP3_output) which is the output csv files for different mlp models with k layers.


if __name__ == '__main__':
    filename = "C:/Users/bssup/Documents/fall22/NLP/hw2/archive/olid-training-v1.0.tsv"
    path_test = "C:/Users/bssup/Documents/fall22/NLP/hw2/archive/testset-levela.tsv"

    model,w2v = train_MLP(filename,1)
    Test_MLP(path_test,model,w2v,1)

    model,w2v = train_MLP(filename,2)
    Test_MLP(path_test,model,w2v,2)

    model,w2v = train_MLP(filename,3)
    Test_MLP(path_test,model,w2v,3)

