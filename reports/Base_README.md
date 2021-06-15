# NLP Product Sentiment Project

**Author:** Jeffrey Hanif Watson

![graph0](./reports/figures/police_car.jpeg)


***
### Quick Links

1. [Final Report](notebooks/report/report.ipynb)
2. [Presentation Slides](reports/presentation.pdf)
***
### Setup Instructions

To setup the project environment, `cd` into the project folder and run `conda env create --file
nlp_project.yml` in your terminal. Next, run `conda activate nlp_project`.
***
## Overview


***
## Business Understanding

***
## Data Understanding


Data set obtained from:
[data.world](https://data.world/crowdflower/brands-and-product-emotions)

***
## Data Preparation
Data cleaning details for the project can be found here:
[Data Cleaning Notebook](notebooks/exploratory/cleaning_eda.ipynb)


***
## Exploring the Stop Data (Highlights From the EDA)

EDA for the project is detailed in the following notebooks:

1. [Initial Analysis Notebook](notebooks/exploratory/eda_visuals.ipynb)

#### General Information


***
## Modeling

### Baseline


![graph6](./reports/figures/Baseline_CM.png)

### Baseline Scores:

#### Score Interpretation

#### Baseline Relative Odds

![graph7](./reports/figures/Baseline_Positive.png)

![graph8](./reports/figures/Baseline_Negative.png)

#### Interpretation of the Odds


### Feature Engineering & Intermediate Models


### Final Model


![graph9](./reports/figures/LR_Final_CM.png)

### Final Scores:

![graph10](./reports/figures/LR_Final_Positive.png)

![graph11](./reports/figures/LR_Final_Negative.png)

#### Relevant Features and Interpretation of Odds


### Final Classifier



![graph12](./reports/figures/RF_Final_CM.png)

### Final Scores:


#### Feature Importances
![graph13](./reports/figures/RF_Final_Feature_Imp.png)


***
## Conclusion

***
## Next Steps


***

## For More Information

Please review our full analysis in [our Jupyter Notebook](./notebooks/report/report.ipynb) or our [presentation](./reports/presentation.pdf).

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
