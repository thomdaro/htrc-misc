from ggplot import *
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import seaborn as sns
import sys

import pyLDAvis.gensim_models
import en_core_web_md

from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore
from gensim.models import CoherenceModel
    

num_topics = 0
df = pd.DataFrame()
corpus = []
dictionary = Dictionary()
tokens = []
data = []
lda_model = None

def preprocess(secure_volume_path, percent=10):
    
    global num_topics, df, corpus, dictionary, tokens, data, lda_model
    
    PERCENTAGE_OF_DATA = percent    

    data = []
    top_dir_name = secure_volume_path
    sub_dir_names = os.listdir(secure_volume_path)
    df = pd.DataFrame(columns=['topic_id', 'topic_weight', 'topic_words', 'doc_id', 'year'])

    for sub_dir_name in sub_dir_names:
        
        dir_name = top_dir_name + '/' + sub_dir_name
        dir = os.fsencode(dir_name)

        for file in os.listdir(dir):
            
            if random.randrange(0, 100) > PERCENTAGE_OF_DATA:
                continue
            
            file_name = os.fsdecode(file)
            page = ''
            
            with open(dir_name + '/' + file_name) as f_open:
                while True:
                    try:
                        line = f_open.readline()
                        if line == '':
                            break
                        page += line
                    except UnicodeDecodeError:
                        continue
                    
            data.append(page)
            
            for n in range(num_topics):
                df.loc[len(df.index)] = [n+1, 0, '', file_name, int(file_name[:4])]

    nlp = en_core_web_md.load()

    removal = ['ADV','PRON','CCONJ','PUNCT','PART','DET','ADP','SPACE', 'NUM', 'SYM']

    tokens = []
    for doc in nlp.pipe(data):
        proj_tok = [token.lemma_.lower() for token in doc if token.pos_ not in removal and not token.is_stop and token.is_alpha]
        tokens.append(proj_tok)

    dictionary = Dictionary(tokens)
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=1000)
    corpus = [dictionary.doc2bow(doc) for doc in tokens]
    
def generate_model():
    
    global num_topics, df, corpus, dictionary, tokens, data, lda_model
    
    lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, iterations=100, num_topics=num_topics, workers=8, passes=100)

    topics = [lda_model[corpus][text] for text in range(len(data))]

    for n in range(len(topics)):
        
        for k in range(num_topics):
            words = ''
            for term in lda_model.show_topic(k, topn=5):
                words += term[0] + ', '
            df.at[n * num_topics + k, 'topic_words'] = words[:-2]
            
        for m in range(len(topics[n])):
            topic = topics[n][m]
            df.at[n * num_topics + topic[0], 'topic_weight'] = topic[1]
         
def visualize(): 
    
    global num_topics, df, corpus, dictionary, tokens, data, lda_model
      
    lda = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(lda, 'images/lda.html')

    df_sum = df[df['topic_weight'] != 0]

    print("Max: {}".format(df_sum['topic_weight'].max()))
    print("Min: {}".format(df_sum['topic_weight'].min()))
    print("Average: {}".format(df_sum['topic_weight'].mean()))
    print("Median: {}".format(df_sum['topic_weight'].median()))
    print("Most frequent value: {}".format(df_sum['topic_weight'].round(3).value_counts().idxmax()))

    p = sns.catplot(x="year", y='topic_weight', hue="topic_id", col='topic_words', col_wrap=2, kind='strip', size=5, aspect=2, jitter=0.1, data=df_sum)
    p.figure.subplots_adjust(top=0.95)
    p.figure.suptitle("Scatterplot of Normalized Topic Weights, Split by Topic; All Weights.", fontsize=16)
    p.savefig("images/scatter-all.png")

    p = create_pointplot(df, 'topic_weight', hue='topic_words', height=5, aspect=1.5, title="Central Range of Normalized Topic Weights. Computed with Seaborn.")
    p.savefig("images/point-all-seaborn.png")

    p = create_pointplot(df, 'topic_weight', col='topic_words', wrap=3, title="Central Range of Normalized Topic Weights Split by Topic. Computed with Seaborn.")
    p.savefig("images/point-sep-seaborn.png")
    
    total_docs = df.groupby('year')['doc_id'].apply(lambda x: len(x.unique())).reset_index()
    total_docs.columns = ['year', 'total_docs']

    df_avg = df.groupby(['year', 'topic_id']).agg({'topic_weight': 'sum'}).reset_index()
    df_avg = df_avg.merge(total_docs, on="year", how="left")
    df_avg['average_weight'] = df_avg['topic_weight'] / df_avg['total_docs']
    df_avg['topic_words'] = df['topic_words']

    p = create_pointplot(df_avg, 'average_weight', hue="topic_words", title="Yearly Average of Normalized Weight per Topic")
    p.savefig("images/point-all-manual.png")

    p = create_pointplot(df_avg, 'average_weight', col="topic_words", wrap=3, title="Yearly Average of Normalized Weight per Topic")
    p.savefig("images/point-sep-manual.png")
    
    p = create_bar(df_avg, 'average_weight', hue="topic_words", title="Yearly Average of Normalized Weight per Topic")
    p.savefig("images/bar-all.png")

    p = create_bar(df_avg, 'average_weight', hue="topic_words", col='topic_words', wrap=3, title="Yearly Average of Normalized Weight per Topic")
    p.savefig("images/bar-sep.png")


def create_pointplot(df, y_value, hue=None, col=None, wrap=None, height=5, aspect=1.5, title=""):
    p = sns.catplot(x="year", y=y_value, kind='point', hue=hue, col=col, col_wrap=wrap, height=height, aspect=aspect, errorbar=('ci', 40), data=df)
    p.figure.subplots_adjust(top=0.9)
    p.figure.suptitle(title, fontsize=16)
    return p


def create_bar(df, y_value, hue=None, col=None, wrap=None, aspect=1.5, title=""):
    p = sns.catplot(x="year", y=y_value, kind='bar', hue=hue, col=col, col_wrap=wrap, aspect=aspect, data=df)
    p.figure.subplots_adjust(top=0.9)
    p.figure.suptitle(title, fontsize=16)
    return p


def coherence(min=1, max=5):
    
    global num_topics, df, corpus, dictionary, tokens, data, lda_model
    
    topics = []
    scores = []

    for i in range(min,max+1):

        lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, iterations=10, num_topics=i, workers=8, passes=10, random_state=100)
        
        cm = CoherenceModel(model=lda_model, texts=tokens, corpus=corpus, dictionary=dictionary, coherence='c_v')
        
        topics.append(i)
        scores.append(cm.get_coherence())

    _=plt.plot(topics, scores)
    _=plt.xlabel('Number of Topics')
    _=plt.ylabel('Coherence Score')
    _=plt.savefig("images/coherence.png")
    
if __name__ == "__main__":
    print("Starting preprocessing...")
    preprocess(sys.argv[1])
    print("Calculating optimal n_topics...")
    coherence()
    print("Generating LDA model...")
    generate_model()
    print("Generating visualizations...")
    visualize()