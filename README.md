# NLP Twitter Sentiment Project

![graph0](./reports/figures/aug_neg.png)

**Author:** Jeffrey Hanif Watson
***
### Quick Links
1. [Data Cleaning/EDA Notebook](./notebooks/exploratory/cleaning_eda.ipynb)
2. [Final Report](./notebooks/report/report.ipynb)
3. [Presentation Slides](./reports/presentation.pdf)
***
### Setup Instructions

To setup the project environment, `cd` into the project folder and run `conda env create --file
nlp_project.yml` in your terminal. Next, run `conda activate nlp_project`.
***
## Overview
In recent years, Twitter has emerged as a prominent platform for marketing and targeted advertising. It is also a valuable conduit for the collection of consumer data, and natural language processing (NLP) can provided solutions to companies seeking to track consumer sentiment regarding their brands and products. This project developed and implemented an NLP model to classify tweets as negative or positive. 

Data cleaning and augmentation, EDA, modeling, and evaluation were performed, and a random forest classifier was chosen as the the final model for the project. Because we wanted to avoid both false positives and false negatives for this project, an accuracy measure of F1 was employed since it is sensitive to both types of error. Since an F1 score is a mix of both precision and recall (F1=1 means perfect recall and precision), interpretation of the results is more easily described in terms of recall and precision. 

An F1 accuracy score of 0.89 (recall=.85, precision=.94) was attained at the end of the project's modeling process. The recall score of .85 meant that 85% of negative tweets were correctly classified as negative, while the precision score of .94 indicated that 94% of tweets classified as negative were truly negative. 

An alternate logistic regression model with an F1 accuracy score of 0.89 (recall=.83, precision=.97) is also available for use by interested parties.
***
## Business Understanding
Companies can benefit from understanding how consumers perceive their brands and products, and sentiment analysis of text data from twitter can help provide this knowledge in a timely manner. A surge in negative sentiment would indicate a crisis of some sort that would need to be addressed quickly, and thus negative sentiment was deemed to be the most important class for modeling purposes in the project. However, data on the level of positive sentiment is still very valuable information, so a model that captures both sentiments as accurately as possible is most desirable.  
***
## Data Understanding
The baseline data for this project consists of a csv file containing 9,093 rows of text data (tweets, brand/product ids, sentiments labels) originally sourced from [Twitter](https://twitter.com/?lang=en) and collected into the [crowdflower/brands-and-product-emotions](https://data.world/crowdflower/brands-and-product-emotions) dataset. From the overwhelming amount of SXSW hashtags, and numerous references to the iPad 2 (which was released on March 2nd, 2011), it appears that the data was collected during the 2011 South by Southwest festival (which ran from March 11th to March 20th). The data contained ternary (positive, neutral, negative) sentiment data which was filtered down to binary (positive, negative) classes for modeling purposes. 

During the modeling process, the class imbalance in the data was shown to interfere with model performance, and additional negative sentiment data was used to augment the baseline data. 1,117 rows of negative general topic tweets were obtained from [Kaggle](https://www.kaggle.com/shashank1558/preprocessed-twitter-tweets), and an additional 1,219 negative Apple tweets were procured from [data.world](https://data.world/crowdflower/apple-twitter-sentiment). This data augmentation greatly improved model performance.
***
## Data Preparation
Data cleaning details for the project can be found here:
[Data Cleaning/EDA Notebook](./notebooks/exploratory/cleaning_eda.ipynb)

A master cleaning function was use to lower case all letters, remove punctuation, urls, retweets, mentions, other unwanted substrings ('{link}', &amp, &quot, &nbsp, &lt, &gt), and return a list of clean and regularized (lemmas and stems) tweets.
***
## Exploring the  Data (Highlights From the EDA)

EDA for the project is detailed in the following notebook:

[Data Cleaning/EDA Notebook](./notebooks/exploratory/cleaning_eda.ipynb)
***
## Emotions by Brand
![graph1](./reports/figures/base_brand_emotions.png)
<font size="4">Apple dominates the tweets with `iPad`, `Apple`, and `iPad or iPhone App` being the subjects of the most tweets.</font>
![graph2](./reports/figures/base_brand_emotions_n.png)
 <font size="4">Tweets about product and brands have a strong postive skew, with the only exception being tweets about the iPhone. `iPhone` has a much more mixed ratio of postives and negatives</font>
***
## Class Balance
The severe class imbalance in the original data was problematic for model development and various resampling techniques were attempted to improve model performance. Ultimately, these methods proved to be unsatisfactory and additional data was found to augment the original dataset, greatly enhancing model performance.

### Original Dataset Class Balance
![graph3](./reports/figures/base_emotion_share.png)

<font size="4">`Neutral` accounts for 60% of the data.</font>

<font size="4">`Positive`                   accounts for 33% of the data.</font>

<font size="4">`Negative`                   accounts for 6% of the data.</font>

Positive and negative tweets are both are under-represented in the data, with negative tweets being extremely under-represented (an order of magnitude less common than the other two classes).

### Augmented Dataset Class Balance
![graph7](./reports/figures/aug_emotion_share.png)

<font size="4">`Neutral` accounts for 48% of the data.</font>

<font size="4">`Positive`                   accounts for 26% of the data.</font>

<font size="4">`Negative`                   accounts for 26% of the data.</font>

The class balance of the augmented data is much more resonable for modeling purposes.

## Word Clouds
Word clouds provide an intuitve understanding of the relative frequencies of words in the data. The tweets were divided into positive, negative, and neutral, and word clouds were formed for each of the emotins. 

### Positive Word Cloud
The size of the word indicates its relative frequency in `Positive` tweets.
![graph8](./reports/figures/aug_pos.png)

<font size="4">`ipad`, and `apple` are the most used words in `Positive` tweets.</font>

### Negative Word Cloud
The size of the word indicates its relative frequency in `Negative` tweets.
![graph9](./reports/figures/aug_neg.png)

<font size="4">`unhappy`, `apple`, and `iphone` are the most used words in `Negative` tweets.</font>

### Neutral Word Cloud
The size of the word indicates its relative frequency in `Neutral` tweets.
![graph10](./reports/figures/aug_neu.png)

<font size="4">`google` and `ipad`, are the most used words in `Neutral` tweets.</font>
***
## Modeling
### Baseline
A baseline model was created from a pipeline consisting of a TFIDF vectorizer and a logistic regression classifier.

### Baseline Scores: F1 = 0.14, Recall = .08, Precision = .85

![graph6](./reports/figures/Baseline_CM.png)

#### Score Interpretation
Since we want to avoid both false positives and false negatives for this project, a metric of F1 was employed because it is sensitive to both types of error. Also, because F1 is a mix of both precision and recall, the interpretation of the results is more easily described in terms of recall and precision. Overall, the performance of the model very poor.
- From the confusion matrix we see that the baseline model is classifying nearly everything as the majority class.
- A recall score of .08 means that 8% of negative tweets were correctly classified as negative. 
- A precision score of .85 indicates that 85% of tweets classified as negative were truly negative.

#### Baseline Features

![graph7](./reports/figures/Baseline_Positive.png)

#### Baseline Relative Odds

![graph7](./reports/figures/Baseline_Positive.png)

![graph8](./reports/figures/Baseline_Negative.png)

#### Interpretation of the Odds
If the assumptions of logistic regression were met by the model, we could numerically quantify the effect of each feature on the model. However, since it is beyond the scope of the project to check that the model meets the underlying assumptions of logistic regression, the most we can say about the features above are their relative importances to the model. A higher bar means more importance of the feature to the model. 

Again, `headache`, `long`, and `fail` are the top features driving `Negative` classifications, while `free`, `new`, and `great` drive `Positive` classifications.

## Data Augmentation & Intermediate Models
The poor performance of the baseline model was largely due the the extreme class imbalance of the original data, so minority class oversampling and SMOTE methods were implemented. These strategies provided improved performance in the baseline model, but the results were still unsatisfactory.  Various other model types were tested with the oversampled data, but the performance of these alternative models was also poor.

In the end, additional negative sentiment data obtained from [Kaggle](https://www.kaggle.com/shashank1558/preprocessed-twitter-tweets) and [data.world](https://data.world/crowdflower/apple-twitter-sentiment) were used to augment the baseline data. This new data greatly imroved the performance of all the models and the final results are detailed below.

## Alternate Model: Tuned Logisitic Regression Classifier

### Scores: F1= 0.89, Recall=.83, Precision=.97 

![graph9](./reports/figures/LR_Final_CM.png)

#### Score Interpretation
From the confusion matrix we see that the model still has a little trouble classifying negatives relative to positives, but the overall performance is acceptable.

- The augmentation of the dataset has greatly improved model performance.
- A recall score of .83 means that 83% of negative tweets were correctly classified as negative. 
- A precision score of .97 indicates that 94% of tweets classified as negative were truly negative.

#### Relevant Features

![graph10](./reports/figures/LR_Final_Positive.png)

![graph11](./reports/figures/LR_Final_Negative.png)

#### Interpretation of the Odds
A higher bar means a greater relative importance of the feature to the model. Again, `Google` and `Ipad` are driving positive classifications, while `unhappy` and `aapl` are driving negative classifications. `Google` is the greatest positve factor followed closely by `Ipad`. `Unhappy` increases the odds of a negative classification most significantly.


## Final Model:  Tuned Random Forest Classifier

![graph12](./reports/figures/RF_Final_CM.png)

### Final Model Scores:
<font size="4">Metrics: F1=0.89, Recall=0.85, Precision=0.94</font>

#### Score Interpretation
From the confusion matrix we see that the MVP model still has a little trouble classifying negatives relative to positives, but the overall performance is acceptable.

- The performance of the baseline model has been greatly improved by the addition of new minority class data. 
- A recall score of .85 means that 85% of negative tweets were correctly classified as negative. 
- A precision score of .94 indicates that 94% of tweets classified as negative were truly negative.

#### Feature Importances
![graph13](./reports/figures/RF_Final_Feature_Imp.png)

#### Notes on the Features

`Ipad` and `unhappy` are again the most important tokens driving the model, and six of the top ten features are brand signifiers.

## Conclusion
A random forest classifier with a F1 accuracy score of 0.89 (recall=.85, precision=.94) was attained at the end of the modeling process and chosen as the final model of the project. The recall score of .85 meant that 85% of negative tweets were correctly classified as negative, and the precision score of .94 indicated that 94% of tweets classified as negative were truly negative. An alternate logistic regression model with an F1 accuracy score of 0.89 (recall=.83, precision=.97) is also available for use by interested parties.

## Next Steps
Next steps for the project include:
- Tuning an XGBoost classifier. 
- Tuning an RNN classifier.
- Implementing a multiclass classifier and adding neutral tweets to the model. 
- Further investigating the logistic regression model's adherence to the underlying assumptions of logistic regression.

## For More Information

Please review our full analysis in our [Jupyter Notebook](./notebooks/report/report.ipynb) or our [presentation](./reports/presentation.pdf).

For any additional questions, please contact **Jeffrey Hanif Watson jeffrey.h.watson@protonmail.com**

## Repository Structure

```
├── README.md
├── data
│   ├── processed
│   └── raw
├── models
├── notebooks
│   ├── exploratory
│   └── report
├── reports
│   └── figures   
└── src
```
