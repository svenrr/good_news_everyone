![Good News](https://motionandmomentum.files.wordpress.com/2017/10/5181a5b8-6e3d-466f-9cea-8f900375fd42-720-000000640b9c2bac.gif)

# Good News Everyone!
**Last update:** 23.11.2020

**Authors:** [Sven Rutsch](https://www.linkedin.com/in/sven-rutsch-b9728612b/), [Fabian März](https://www.linkedin.com/in/fabian-m%C3%A4rz-3913981b3/) & [Christoph Blickle](https://www.linkedin.com/in/christoph-blickle-4064ab1ba)

This is a project in which we try to differentiate between positive and negative/neutral news from the web automatically. This is done with an accuracy of 85%.
For this we use a variety of machine learning models and dictionary based apporaches to sentiment analysis. 
Furthermore the goal of this project was to categorize (e.g. business, finance, sports etc.) and summarize news articles.
The finished product could either be a website on which users could find only positive news, or a livefeed, where our project could serve as a helpful tool for journalists or similar professions.

We have adapted and optimized some scripts for our interactive [WebApp](https://share.streamlit.io/svenrr/gne-webapp-streamlit/main/main.py). We created a separate repository for this, which Streamlit.io accesses. You can find it [here](https://github.com/svenrr/GNE-webapp-streamlit).

**Short presentation:** https://docs.google.com/presentation/d/1cOYTSatxMXIUgyhFldnGQPYo4UV8NU7sPX3bwIPU4xI/edit?usp=sharing


## Table of Content

[1. Data acquistion and preprocessing](#data-acqusition-and-preprocessing)

[2. Sentiment Analysis](#sentiment-analysis)

[3. Categorization](#categorization)

[4. Summarization](#summarization)

[5. Webapp](#webapp)


### Data acqusition and preprocessing

The main source of the data is webhose.io, a database for web content. From there we downloaded ~400k articles. After extensive data cleaning we ended up at ~70k articles. 
These articles were already categorized (finance, politics etc.), but no label for the sentiment was given. We labeled part of the data by running 3 different dictionary based sentiment analysis. We chose three thresholds which made sure that an article was in fact positive or at least neutral. Neutral and bad articles were classified as one category.
For more reliable data to add, we used positive news websites and scraped articles from there. We also used some subreddits as sources for good and bad news articles. In the end we assemble a dataset with ~7k labeled articles for supervised machine learning.

The preprocessing of the articles was mainly done with spaCy, which we used for the tokenization. To vectorize the remaining data we used the TfidfVectorizer from Scikit-learn.
These two processing methods and tools gave the best result out of all the libraries we tried (e.g. Gensim tokenizing and Word2Vec-Transformation)

### Sentiment Analysis

For sentiment analysis we trained different machine learning algorithms like LogisticRegression or NaiveBayes on the labeled dataset.
A VotingClassifier serves as the final decision maker and has an accuracy of around 85% at this point in time.

### Categorization

Different machine learning algorithms were trained to categorize articles into seven categories.
The metric used was accuracy, because the dataset was balanced, and each category had the same importance.
The final models reach an accuracy over 80%. You can learn more about it [here](Category_Analysis/README.md)

### Summarization

We perform an extractive summarization by looking at the different word frequencies. These are then normalized and weighted differently. Accordingly, the different sentences can then be weighted and ranked. By default, the top 10% of sentences are used to create the summary.

### Webapp & Deployment 

Streamlit was used to create the [Webapp](https://share.streamlit.io/svenrr/gne-webapp-streamlit/main/main.py). The site offers the possibilty to copy and paste a self chosen article and see how well the algorithms mentioned above perform on it.
Furthermore you can finde a preview of our idea for a live news feed, as well as some EDA.

![Deployment](https://snipboard.io/8FGKri.jpg)


