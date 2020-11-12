![Good News](https://motionandmomentum.files.wordpress.com/2017/10/5181a5b8-6e3d-466f-9cea-8f900375fd42-720-000000640b9c2bac.gif)

=====================
# Good News Everyone!

This is a project in which we try to differentiate between positive and negative/neutral news from the web automatically.
For this we use a variety of machine learning models and dictionary based apporaches to sentiment analysis. 
Furthermore the goal of this project was to categorize (e.g. business, finance, sports etc.) and summarize news articles.
The finished product could either be a website on which users could find only positive news, or a livefeed, where our project could serve as a helpful tool for journalists or similar professions.

=====================

## Table of Content
---------------------

[1. Overview](#overview)

[2. Data acquistion](#data-acqusition-&-engineering)

[3. Sentiment Analysis](#sentiment-analysis)

[4. Categorization](#categorization)

[5. Summarization](#summarization)

[6. Webapp](#webapp)

---------------------

### Overview




### Data acqusition & engineering

The main source of the data is webhose.io, a database for web content. From there we loaded ~400k articles. After extensive data cleaning we ended at ~70k articles. 
These articles were already categorized (finance, politics etc.), but no label for the sentiment was given. We labeled part of the data by running 3 different dictionary based sentiment analysis. We chose three thresholds which made sure that an article was in fact positive or at least neutral. Neutral and bad articles were classified as one category.
For more reliable data to add, we used positive news websites and scraped articles from there. We also used some subreddits as sources for good and bad news articles. In the end we assemble a dataset with ~7k labeled articles for supervised machine learning.
PREPROCESSING

### Sentiment Analysis

For sentiment analysis we trained standard machine learning algorithms like SVM or LogisticRegression on the labeled dataset.
A VotingClassifier serves as the final decision maker and has an accuracy of around 86% at this point in time.

### Categorization

TO BE FILLED

### Summarization

TO BE FILLED

### Webapp

For the Webapp we used streamlit.
