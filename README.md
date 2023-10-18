# Twitter Sentiment Analysis Project

## Overview
This project focuses on performing sentiment analysis on Twitter comments using Natural Language Processing (NLP) techniques. It aims to determine whether a given comment conveys a positive or negative sentiment. The project is implemented in Python and utilizes various libraries including pandas, numpy, seaborn, wordcloud, nltk, re, matplotlib, plotly, and scikit-learn.

## Prerequisites
- Python 3.x
- Required Libraries:
  - pandas
  - numpy
  - seaborn
  - wordcloud
  - nltk
  - matplotlib
  - plotly
  - scikit-learn

To install the necessary libraries, use the following command:

```bash
pip install pandas numpy seaborn wordcloud nltk matplotlib plotly scikit-learn
```

Additionally, download the NLTK stopwords dataset using the following Python code:

```python
import nltk
nltk.download('stopwords')
```

## System Requirements
- A minimum of 11 GB RAM is recommended for smooth execution.

![Screenshot_20231010_125826](https://github.com/kushagra8881/twitter_sentiment_analysis/assets/127012998/40260c35-60c2-478d-a735-fdd926673b32)

## Positive Wordcloud
![Positive Wordcloud](https://github.com/kushagra8881/twitter_sentiment_analysis/assets/127012998/243d1493-c747-4b48-9b35-c06431f9f5f5)

## Project Structure
```
twitter_sentiment_analysis/
│
├── data/                       # Directory for storing the dataset
│   └── twitter_comments.csv     # Sample Twitter comments dataset
│
├── output/                     # Directory for storing output files and images
│
├── sentiment_analysis.ipynb     # Jupyter Notebook for sentiment analysis
│
└── README.md                   # Project documentation
```

## Running the Project
1. Ensure that all prerequisites are met, and the dataset (`twitter_comments.csv`) is placed in the `data/` directory.

2. Open the `sentiment_analysis.ipynb` notebook using Jupyter.

3. Execute each cell in the notebook sequentially to perform tasks including:
   - Data preprocessing
   - Exploratory Data Analysis (EDA)
   - Text processing (removing stopwords, stemming, etc.)
   - Visualizations (WordCloud, Bar charts, etc.)
   - Model training (Naive Bayes, Support Vector Machine)
   - Evaluating model performance

4. The notebook will generate a visualization of positive and normal comments, saved in the `output/` directory.

5. After running the notebook, you'll be able to make predictions on new comments using the trained model.

## Making Predictions Online
- The project has been extended to include an online prediction feature using Flask.
- A pickled version of the trained model (`sentiment_model.pkl`) is available.
- Send a POST request to `http://127.0.0.1:5000/predict` with a JSON object containing the comment you want to analyze.

### Example using Python requests:
![Screenshot_20231018_093448](https://github.com/kushagra8881/twitter_sentiment_analysis/assets/127012998/9dd4e945-79c8-47f1-9ba0-56e6cbb62014)

```python
import requests

url = 'http://127.0.0.1:5000/predict'
data = {'comment': 'This is a positive comment.'}
response = requests.post(url, json=data)
result = response.json()
print(result)
```

## Note
- Ensure you have ample storage space for saving the output images and model files.
- The model training phase may require a significant amount of memory, so it's recommended to have a minimum of 11 GB RAM available.

For any issues or inquiries, please refer to the project documentation or contact the project maintainers.

Happy analyzing!
