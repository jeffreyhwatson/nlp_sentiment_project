import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def brand_freqs(df):
    brands = df.brand_product.value_counts(normalize=True)
    brands_df = pd.DataFrame(brands)
    brands_df.reset_index(inplace=True)
    brands_df.columns = ['Brand/Product', 'Percentage']

    fig, ax = plt.subplots(figsize=(20,8))
    sns.barplot(x='Percentage', y='Brand/Product', edgecolor='deepskyblue', palette='Blues_r', data=brands_df)
    ax.tick_params(labelsize=20)
    plt.title('Brand Share of Data', fontsize=30)
    plt.xlabel('', fontsize=20)
    plt.ylabel("")
    # plt.savefig('brand_freqs',  bbox_inches ="tight",\
    #             pad_inches = .25, transparent = False)
    plt.show()
    
def emotion_freqs(df):
    emotion = df.emotion.value_counts(normalize=True)
    emotion_df = pd.DataFrame(emotion)
    emotion_df.reset_index(inplace=True)
    emotion_df.columns = ['Emotion', 'Share']

    fig, ax = plt.subplots(figsize=(20,8))
    sns.barplot(x='Share', y='Emotion', edgecolor='deepskyblue', palette='Blues_r', data=emotion_df)
    ax.tick_params(labelsize=20)
    plt.title('Emotion Share of Data', fontsize=30)
    plt.xlabel('', fontsize=20)
    plt.ylabel("")
    # plt.savefig('emotion_share',  bbox_inches ="tight",\
    #             pad_inches = .25, transparent = False)
    plt.show()
    
def null_brand_emotions(df):
    null_brand_emotion = df[(df['brand_product'].isna()) &\
     (df['emotion'] != 'No emotion toward brand or product' )]
    emotion = null_brand_emotion.emotion.value_counts()
    emotion_df = pd.DataFrame(emotion)
    emotion_df.reset_index(inplace=True)
    emotion_df.columns = ['Emotion', 'Count']

    fig, ax = plt.subplots(figsize=(20,8))
    sns.barplot(x='Count', y='Emotion', edgecolor='deepskyblue',
                palette='Blues_r', data=emotion_df)
    ax.tick_params(labelsize=20)
    plt.title('Null Brand Emotion Counts', fontsize=30)
    plt.xlabel('', fontsize=20)
    plt.ylabel("")
    # plt.savefig('null_emotion_counts',  bbox_inches ="tight",\
    #             pad_inches = .25, transparent = False)
    plt.show()
    
def brand_emotions(df):
    bdf = df.groupby('brand_product')['emotion'].value_counts(normalize=True).unstack()
    bdf.plot(kind='barh', figsize=(20,10), color=['ghostwhite', 'lightblue', 'steelblue', 'darkblue'], edgecolor='deepskyblue')
    
    plt.xticks(rotation=0)
    plt.title('Emotions by Brand/Product', fontsize=30)
    plt.xlabel('')
    plt.ylabel('')
    plt.legend(title='Emotion',
               labels=['Unknown', 'Positive', 'Negative', 'Neutral'])
    # plt.savefig('brand_emotions',  bbox_inches ="tight",\
    #             pad_inches = .25, transparent = False)
    plt.show()

def hashtag_c(df):
    counts = df['hashtags'].value_counts()[:20]
    percents = df['hashtags'].value_counts(normalize=True)[:20]
    tags = df['hashtags'].value_counts()[:20].index

    fig, ax = plt.subplots(figsize=(16,8))
    sns.barplot(x=counts, y=tags, palette='Blues_r')
    plt.title('Counts of the Top 20 Hashtags')
    plt.xlabel('Count')
    # plt.savefig('brand_emotions',  bbox_inches ="tight",\
    #             pad_inches = .25, transparent = False)
    plt.show()

def hashtag_p(df):
    counts = df['hashtags'].value_counts()[:20]
    percents = df['hashtags'].value_counts(normalize=True)[:20]
    tags = df['hashtags'].value_counts()[:20].index

    fig, ax = plt.subplots(figsize=(16,8))
    sns.barplot(x=percents, y=tags, palette='Blues_r')
    plt.title('Percentages of the Top 20 Hashtags')
    plt.xlabel('Percent')
    # plt.savefig('brand_emotions',  bbox_inches ="tight",\
    #             pad_inches = .25, transparent = False)
    plt.show()
    
    
def top_word_list(data, n):
    "returns the n most common words in a corpus."
    processed_data = list(map(tokens, data))
    word_li = word_list(processed_data)
    freqdist = FreqDist(word_li)
    most_common = freqdist.most_common(n)
    top_word_list = [tup[0] for tup in most_common]
    plt.figure(figsize=(15,7))
    freqdist.plot(n)  
    # plt.savefig('brand_emotions',  bbox_inches ="tight",\
    #             pad_inches = .25, transparent = False)
    print(top_word_list)