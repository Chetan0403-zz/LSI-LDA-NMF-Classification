import numpy as np
import pandas as pd
import gc
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import re
from contextlib import contextmanager
import nltk
import time
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD

@contextmanager
def timer(name):
    """
    Taken from Konstantin Lopuhin https://www.kaggle.com/lopuhin
    in script named : Mercari Golf: 0.3875 CV in 75 LOC, 1900 s
    https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
    """
    t0 = time.time()
    yield
    #print(f'[{name}] done in {time.time() - t0:.0f} s')
    print("(%s) done in %.0f s" % (name, time.time()-t0)) # Activate on GPU

def clean_text(text):
    """
    Does basic text cleaning before feeding into vectorizer
    """
    # Replace Emoticons
    text = text.replace(":/", " bad ")
    text = text.replace(":&gt;", " sad ")
    text = text.replace(":')", " sad ")
    text = text.replace(":-(", " frown ")
    text = text.replace(":(", " frown ")
    text = text.replace(":s", " frown ")
    text = text.replace(":-s", " frown ")
    text = text.replace("&lt;3", " heart ")
    text = text.replace(":d", " smile ")
    text = text.replace(":p", " smile ")
    text = text.replace(":dd", " smile ")
    text = text.replace("8)", " smile ")
    text = text.replace(":-)", " smile ")
    text = text.replace(":)", " smile ")
    text = text.replace(";)", " smile ")
    text = text.replace("(-:", " smile ")
    text = text.replace("(:", " smile ")
    text = text.replace(":/", " worry ")
    text = text.replace(":&gt;", " angry ")
    text = text.replace(":')", " sad ")
    text = text.replace(":-(", " sad ")
    text = text.replace(":(", " sad ")
    text = text.replace(":s", " sad ")
    text = text.replace(":-s", " sad ")
    
    # Replace Shortforms 
    text = re.sub(r'[\w]*don\'t[\w]*','do not',text)
    text = re.sub(r'[\w]*i\'ll[\w]*','i will',text)
    text = re.sub(r'[\w]*wasn\'t[\w]*','was not',text)
    text = re.sub(r'[\w]*there\'s[\w]*','there is',text)
    text = re.sub(r'[\w]*i\'m[\w]*','i am',text)
    text = re.sub(r'[\w]*won\'t[\w]*','will not',text)
    text = re.sub(r'[\w]*let\'s[\w]*','let us',text)
    text = re.sub(r'[\w]*i\'d[\w]*','i would',text)
    text = re.sub(r'[\w]*they\'re[\w]*','they are',text)
    text = re.sub(r'[\w]*haven\'t[\w]*','have not',text)
    text = re.sub(r'[\w]*that\'s[\w]*','that is',text)
    text = re.sub(r'[\w]*couldn\'t[\w]*','could not',text)
    text = re.sub(r'[\w]*aren\'t[\w]*','are not',text)
    text = re.sub(r'[\w]*wouldn\'t[\w]*','would not',text)
    text = re.sub(r'[\w]*you\'ve[\w]*','you have',text)
    text = re.sub(r'[\w]*you\'ll[\w]*','you will',text)
    text = re.sub(r'[\w]*what\'s[\w]*','what is',text)
    text = re.sub(r'[\w]*we\'re[\w]*','we are',text)
    text = re.sub(r'[\w]*doesn\'t[\w]*','does not',text)
    text = re.sub(r'[\w]*can\'t[\w]*','can not',text)
    text = re.sub(r'[\w]*shouldn\'t[\w]*','should not',text)
    text = re.sub(r'[\w]*didn\'t[\w]*','did not',text)
    text = re.sub(r'[\w]*here\'s[\w]*','here is',text)
    text = re.sub(r'[\w]*you\'d[\w]*','you would',text)
    text = re.sub(r'[\w]*he\'s[\w]*','he is',text)
    text = re.sub(r'[\w]*she\'s[\w]*','she is',text)
    text = re.sub(r'[\w]*weren\'t[\w]*','were not',text)
    
    # Remove punct except ! and ?
    text = re.sub(r"[,.:|(;@)-/^—#&%$<=>`~{}\[\]\'\"]+\ *", " ", text)
    
    # Separate out ! and ?
    text = re.sub("!", " !", text)
    text = re.sub("\?", " ?", text)
    
    # Remove numbers
    text = re.sub("\\d+", " ", text)
    
    # Remove additional space
    text = ' '.join(text.split())
    return text


if __name__ == '__main__':
    
    with timer("Loading training, testing data"):
        train = pd.read_csv("train.csv")
        test = pd.read_csv("test.csv")
        label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        y = train[label_cols].values
    
    with timer("Cleaning text"):
        train['comment_text'] = train['comment_text'].apply(lambda x: clean_text(x))
        test['comment_text'] = test['comment_text'].apply(lambda x: clean_text(x))
        
        train_text = train['comment_text']
        test_text = test['comment_text']
        all_text = pd.concat([train_text, test_text])
           
    with timer("Fitting word vectorizers"):      
        word_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            min_df=3, max_df=0.7,
            #stop_words='english',
            ngram_range=(1, 1),
            max_features=50000)
        word_vectorizer.fit(all_text)
                
    with timer("Transforming train into vectors"):
        train_word_features = word_vectorizer.transform(train_text)

    with timer("Extract top 20 features with LSI, LDA, NMF"):          
        # Build a Latent Semantic Indexing Model 
        lsi_model = TruncatedSVD(20) #Number of components to extract from documents
        lsi = lsi_model.fit_transform(train_word_features)
              
        # Build a Latent Dirichlet Allocation Model
        lda_model = LatentDirichletAllocation(n_topics=20, max_iter=10, learning_method='online')
        lda = lda_model.fit_transform(train_word_features)
        
        # Build a Non-Negative Matrix Factorization Model
        nmf_model = NMF(n_components=20)
        nmf = nmf_model.fit_transform(train_word_features)
        
        # Printing top 20 topics
        """
        Picked from https://nlpforhackers.io/topic-modeling/
        """
        def print_topics(model, vectorizer, top_n=20):
            for idx, topic in enumerate(model.components_):
                print("Topic %d:" % (idx))
                print([(vectorizer.get_feature_names()[i], topic[i])
                                for i in topic.argsort()[:-top_n - 1:-1]])
 
        print("LDA Model:")
        print_topics(lda_model, word_vectorizer)
        print("=" * 20)
         
        print("NMF Model:")
        print_topics(nmf_model, word_vectorizer)
        print("=" * 20)
         
        print("LSI Model:")
        print_topics(lsi_model, word_vectorizer)
        print("=" * 20)
    
    with timer("Running logistic regression on NMF, LDA, LSI features"):
        
        for df,df_name in zip([lda,lsi,nmf],["LDA","LSI","NMF"]):    
            
            print("\n Running Logreg on data from {}\n".format(df_name))
            scores = []
            classifier = LogisticRegression(solver='sag')
            
            for i,label in enumerate(label_cols):
                score = cross_val_score(classifier, df, train[label], cv=5, scoring='roc_auc')
                print("\t {} CV AUC: {}".format(label, round(np.mean(score),4)))
                scores.append(np.mean(score))
             
            print("\n {} Mean CV AUC score: {}".format(df_name,round(np.mean(scores),4)))