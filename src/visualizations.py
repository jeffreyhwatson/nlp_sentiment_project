import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def brand_counts(df):
    brands = df.brand_product.value_counts()
    brands_df = pd.DataFrame(brands)
    brands_df.reset_index(inplace=True)
    brands_df.columns = ['Brand/Product', 'Count']

    fig, ax = plt.subplots(figsize=(20,8))
    sns.barplot(x='Count', y='Brand/Product', edgecolor='deepskyblue', palette='Blues_r', data=brands_df)
    ax.tick_params(labelsize=20)
    plt.title('Brand Counts', fontsize=30)
    plt.xlabel('', fontsize=20)
    plt.ylabel("")
    # plt.savefig('brand_counts',  bbox_inches ="tight",\
    #             pad_inches = .25, transparent = False)
    plt.show()
    
def emotion_counts(df):
    emotion = df.emotion.value_counts()
    emotion_df = pd.DataFrame(emotion)
    emotion_df.reset_index(inplace=True)
    emotion_df.columns = ['Emotion', 'Count']

    fig, ax = plt.subplots(figsize=(20,8))
    sns.barplot(x='Count', y='Emotion', edgecolor='deepskyblue', palette='Blues_r', data=emotion_df)
    ax.tick_params(labelsize=20)
    plt.title('Emotion Counts', fontsize=30)
    plt.xlabel('', fontsize=20)
    plt.ylabel("")
    # plt.savefig('emotion_counts',  bbox_inches ="tight",\
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