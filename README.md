# Product Sentiment Analysis Project

![graph0](./reports/figures/aug_neg.png)

**Author:** Jeffrey Hanif Watson
***
### Quick Links
1. [Data Cleaning/EDA Notebook](./notebooks/exploratory/cleaning_eda.ipynb)
2. [Final Report](./notebooks/report/report.ipynb)
3. [Presentation Slides](./reports/presentation.pdf)
***
### Repository Structure

```
├── README.md
├── nlp_project.yml
├── data
│   ├── processed
│   └── raw
├── models
├── notebooks
│   ├── exploratory
│   └── report
├── reports
│   └── figures
└── src  
```
***
### Setup Instructions
To setup the project environment, `cd` into the project folder and run `conda env create --file
nlp_project.yml` in your terminal. Next, run `conda activate nlp_project`.
***
## Overview:
In recent years, Twitter has emerged as a prominent platform for marketing and targeted advertising. It is also a valuable conduit for the collection of consumer data, and natural language processing (NLP) methods can provide a solution for companies seeking to track consumer sentiment with respect to their brands and products. This project developed and implemented several NLP models to classify tweets as either negative or positive. 

Data cleaning, EDA, modeling, and evaluation were performed, and a logistic regression model with an F1 accuracy score of 0.89 (recall=.83, precision=.97) was chosen as the the final model for the project. Because we wanted to avoid both false positives and false negatives for this project, an accuracy measure of F1 was employed since it is sensitive to both types of error. An F1 score is a mix of both precision and recall (F1=1 means perfect recall and precision), so interpretation of the results is more easily described in terms of recall and precision. 

An F1 accuracy score of 0.89 (recall=.83, precision=.97) was attained at the end of the project's modeling process. The recall score of .83 meant that 83% of negative tweets were correctly classified as negative, while the precision score of .97 indicated that 97% of tweets classified as negative were truly negative. 

An alternate random forest classifier model with an F1 accuracy score of 0.89 (recall=.85, precision=.94) is also available for use by interested parties.
***
## Business Understanding
Companies can benefit from understanding how consumers perceive their brands and products, and sentiment analysis of text data from twitter can help provide this knowledge in a timely manner. A surge in negative sentiment would indicate a crisis of some sort that would need to be addressed quickly, and thus negative sentiment was deemed to be the most important class for modeling purposes in the project. However, data on the level of positive sentiment is still very valuable information for strategic planning and building on past successes, so a model that captures both sentiments as accurately as possible is most desirable.  
***
## Data Understanding
A data frame was formed from a csv file containing 9,093 rows of text data (tweets, brand/product ids, sentiment labels) originally sourced from [Twitter](https://twitter.com/?lang=en) and collected into the [crowdflower/brands-and-product-emotions](https://data.world/crowdflower/brands-and-product-emotions) dataset. From the overwhelming amount of SXSW hashtags, and numerous references to the iPad 2 (which was released on March 2nd, 2011), it appears that the data was collected during the 2011 South by Southwest festival (which ran from March 11th to March 20th). The data contained ternary (positive, neutral, negative) sentiment data which was filtered down to binary (positive, negative) classes for modeling purposes. 

During the modeling process, the class imbalance in the data was shown to interfere with model performance, and additional negative sentiment data was used to augment the original data set. 1,117 rows of negative general topic tweets were obtained from [Kaggle](https://www.kaggle.com/shashank1558/preprocessed-twitter-tweets), and an additional 1,219 negative Apple tweets were procured from [data.world](https://data.world/crowdflower/apple-twitter-sentiment). The combined data sets resulted in 11,242 rows of data, and this data augmentation greatly improved model performance.
***
## Data Preparation
Data cleaning details for the project can be found here:
[Data Cleaning/EDA Notebook](./notebooks/exploratory/cleaning_eda.ipynb)

Master cleaning functions were used to lower case all letters, remove punctuation, urls, retweets, mentions, other unwanted substrings ('{link}', &amp, &quot, &nbsp, &lt, &gt), and return a list of clean and regularized (lemmas and stems) tweets.

Lemmatized data was used during the majority of the modeling process, but the final models were also tuned and tested on stemmed data to see if there were any boosts to model performance. Ultimately, the models performed roughly the same on stemmed and lemmatized data, and we considered lemmas preferable due to their greater (human) readability when presenting the project's EDA and interpreting the features driving the models

Further, while the set of stems had about one thousand fewer tokens, and lemmatization is generally considered slower than stemming, lemmatization and modeling with lemmatized data didn't raise any significant speed or memory issues when compared to working with stemmed data.
***
# Exploring the  Data (Highlights From the EDA)

EDA for the project is detailed in the following notebook: [Data Cleaning/EDA Notebook](./notebooks/exploratory/cleaning_eda.ipynb)
***
## Class Balance
The severe class imbalance in the original data was problematic for model development and various resampling techniques were attempted to improve model performance. Ultimately, these methods proved to be unsatisfactory and additional data was found to augment the original dataset, greatly enhancing model performance.

### Original Dataset Class Balance
![graph3](./reports/figures/base_emotion_share.png)

<font size="4">`Neutral` accounts for 60% of the data.</font>

<font size="4">`Positive`                   accounts for 33% of the data.</font>

<font size="4">`Negative`                   accounts for 6% of the data.</font>

Positive and negative tweets are both under-represented in the original data, with negative tweets being extremely under-represented.

### Augmented Dataset Class Balance
![graph4](./reports/figures/aug_emotion_share.png)

<font size="4">`Neutral` accounts for 48% of the data.</font>

<font size="4">`Positive`                   accounts for 26% of the data.</font>

<font size="4">`Negative`                   accounts for 26% of the data.</font>

The class balance of the augmented data is much more reasonable for modeling purposes.
***
## Emotions by Brand
The vast majority of neutral tweets had null values in the `brand_product` column. After filtering the data down to tweets that had brand/product information attached to them, we visualized the level of each sentiment for the various brands/products.

![graph1](./reports/figures/base_brand_emotions.png)
<font size="4">Apple dominates the tweets with `iPad`, `Apple`, and `iPad or iPhone App` being the subjects of the most tweets.</font>

![graph2](./reports/figures/base_brand_emotions_n.png)
 <font size="4">Tweets about product and brands have a strong positive skew, with the only exception being tweets with `iPhone` values. `iPhone` has a much higher ratio of negative to positives tweets than the other products and brands.</font>
***
## Word Clouds
Word clouds help to display the relative frequencies of words in the data in an intuitive way. The tweets were divided into positive, negative, and neutral, and word clouds were formed for each of the emotions. 

### Positive Word Cloud
The size of the word indicates its relative frequency in `Positive` tweets.
![graph5](./reports/figures/aug_pos.png)

<font size="4">`ipad`, and `apple` are the most used words in `Positive` tweets.</font>

### Negative Word Cloud
The size of the word indicates its relative frequency in `Negative` tweets.
![graph6](./reports/figures/aug_neg.png)

<font size="4">`unhappy`, `apple`, and `iphone` are the most used words in `Negative` tweets.</font>

### Neutral Word Cloud
The size of the word indicates its relative frequency in `Neutral` tweets.
![graph7](./reports/figures/aug_neu.png)

<font size="4">`google`, `apple`, and `ipad` are the most used words in `Neutral` tweets.</font>
***
# Modeling
The data was filtered down to negative and positive tweets, and various binary classifiers were trained and tested during the modeling process. The results of these experiments are detailed below.

Details of the full modeling process can be found here:
[Modeling Notebook](./notebooks/exploratory/modeling_eda.ipynb)


## Baseline Model:
A baseline model was created from a pipeline consisting of a TFIDF vectorizer and a dummy classifier.

![graph8](./reports/figures/dummy.png)

<font size="4">Baseline Scores: F1 = 0, Recall = 0, Precision = 0</font>

#### Score Interpretation
F1 is a mix of both precision and recall, so the interpretation of the results is more easily given in terms of recall and precision. 
- From the confusion matrix we see that the baseline model is classifying everything as the majority class, which was expected.
- No tweets were correctly classified as negative, so the recall score for this model is 0. 
- No tweets were classified as negative, so the precision score (the proportion of tweets classified as negative that were truly negative) is 0 as well.
***
## First Simple Model:

<font size="4">Average Validation Scores: F1=.07, Recall=.04, Precision=.95</font>

A first simple model was created from a pipeline consisting of a TFIDF vectorizer and a logistic regression classifier. Logistic regression was chosen as the first model because of its speed and ease of interpretability.


![graph18](./reports/figures/baseline_cm.png)

<font size="4">Scores on Test Data: F1 = .14, Recall = .08, Precision = .85</font>

### Score Interpretation
Since F1 is a mix of both precision and recall, the interpretation of the results is more easily described in terms of recall and precision. 
- From the confusion matrix we see that the simple model is classifying nearly everything as the majority class.
- A recall score of .08 means that 8% of negative tweets were correctly classified as negative. 
- A precision score of .85 indicates that 85% of tweets classified as negative were truly negative.

While it was a slight improvement over the baseline model's, the performance of the first simple model was still very poor. It was only capturing 8% of our desired class, and that is insufficient for use in our business case.
***
#### Data Augmentation & Intermediate Models
The poor performance of the first simple model was largely due the the extreme class imbalance of the original data, so minority class oversampling and SMOTE methods were implemented. These strategies provided improved performance in the simple model, but the results were still unsatisfactory.  Various other model types were tested with the oversampled data, but the performance of these alternative models was also poor.

In the end, additional negative sentiment data obtained from [Kaggle](https://www.kaggle.com/shashank1558/preprocessed-twitter-tweets) and [data.world](https://data.world/crowdflower/apple-twitter-sentiment) were used to augment the baseline data. This new data greatly improved the performance of all the models. The average metrics and details of some of the intermediate models are detailed below: 

- Simple logistic regression: F1=.89, Recall=.84 Precision=.94 (Untuned)

- Tuned Naive Bayes classifier: F1=.87, Recall=.81 Precision=.94 (Tuned with GridSearchCV)

- Tuned XGBoost Classifier: F1=.88, Recall=.86, Precision=.91 (Tuned with RandomizedSearchCV)

Ultimately, the final model slightly improved on the metrics of the untuned logistic regression, naive Bayes classifier, and XGBoost Classifier to varying degrees. The performance of the XGBoost model was close to that of the final model, but came at the expense of considerably longer training and tuning times, a higher computational cost, and less interpretability.
***
## Final Model:
<font size="4">Logistic Regression CLF Tuned on Augmented Lemmatized Data</font>

<font size="4">Average Scores: F1=.90, Recall=.84, Precision=.96</font>

A TFIDF vectorizer was used for feature engineering and vectorization.

Given its overall performance, the tuned logistic regression model is the final choice for this project due to its greater training, tuning, and prediction speeds, as well as its lower computational cost.

![graph12](./reports/figures/tuned_logreg_cm.png)

<font size="4">Scores on Test Data: F1=.89, Recall=.83, Precision=.97</font>

#### Score Interpretation
From the confusion matrix we see that the model still has a little more trouble classifying negatives relative to positives, but the overall performance is acceptable.

- The augmentation of the dataset has greatly improved model performance.
- A recall score of .83 means that 83% of negative tweets were correctly classified as negative. 
- A precision score of .97 indicates that 97% of tweets classified as negative were truly negative.

#### Feature Coefficients
![graph13](./reports/figures/tuned_coeff.png)

#### Notes on the Features

`google` & `ipad` have the largest coefficients driving positive classifications, while `unhappy` and `aapl` (Apple's stock symbol) have the largest coefficients driving negative classifications. Most brand signifiers are still associated with positive classifications.

#### Relative Odds
![graph14](./reports/figures/tuned_negative.png)

![graph15](./reports/figures/tuned_positive.png)

#### Interpretation of the Odds
A higher bar means a greater relative importance of the feature to the model. Again, google and ipad are the greatest factors driving positive classifications, while unhappy and aapl are driving negative classifications. unhappy increases the odds of a negative classification most significantly.

## Alternate Final Model:  
<font size="4">Tuned Random Forest Classifier</font>

<font size="4">Average Scores: F1=.89, Recall=.85, Precision=.94</font>

A TFIDF vectorizer was used for feature engineering and vectorization.

The performance of the random forest model is comparable  to that of the the logistic regression model, albeit with slightly higher recall and slightly lower precision. However, it is slower and more computationally expensive. If the highest possible recall is needed, and speed and computational power are of minimal concern, then this model might be preferable.

![graph16](./reports/figures/tuned_rf_cm.png)

<font size="4">Scores on Test Data: F1=.89, Recall=.85, Precision=.94</font>

#### Score Interpretation
From the confusion matrix we see that the model still has a little more trouble classifying negatives relative to positives, but the overall performance is acceptable.

- A recall score of .85 means that 85% of negative tweets were correctly classified as negative. 
- A precision score of .94 indicates that 94% of tweets classified as negative were truly negative.

#### Feature Importances
![graph17](./reports/figures/feature_imp.png)

#### Notes on the Features

`ipad` and `unhappy` are again the most important words driving the model, and six of the top ten features are brand or product signifiers.

# Conclusion
A tuned logistic regression model was chosen as the final model of the project, and an F1 accuracy score of .89 (recall=.83, precision=.97) achieved at the end of the modeling process. The recall score of .83 meant that 83% of negative tweets were correctly classified as negative, and the precision score of .97 indicated that 97% of tweets classified as negative were truly negative. 

An alternate random forest classifier with a F1 accuracy score of .89 (recall=.85, precision=.94) is also available for use if a higher recall is the most pressing concern.

# Next Steps
Next steps for the project include: 
- Using an advanced word embedding method and tuning an RNN classifier.
- Implementing a multiclass classifier and adding neutral tweets to the model. 
- Further investigating the final model's adherence to the underlying assumptions of logistic regression.

# For More Information

Please review our full analysis in our [Jupyter Notebook](./notebooks/report/report.ipynb) or our [presentation](./reports/presentation.pdf).

For any additional questions, please contact **Jeffrey Hanif Watson jeffrey.h.watson@protonmail.com**
