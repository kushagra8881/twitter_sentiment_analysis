# twitter_sentiment_analysis
# Twitter Sentiment Analysis Project

## Overview
This project aims to perform sentiment analysis on Twitter comments. It involves using Natural Language Processing (NLP) techniques to determine whether a given comment is positive or negative. The project uses Python and several libraries including pandas, numpy, seaborn, wordcloud, nltk, re, matplotlib, wordcloud, plotly, and scikit-learn.

## Prerequisites
- Python 3.x
- Libraries:
  - pandas
  - numpy
  - seaborn
  - wordcloud
  - nltk
  - matplotlib
  - plotly
  - scikit-learn

Install the required libraries using the following:
```bash
pip install pandas numpy seaborn wordcloud nltk matplotlib plotly scikit-learn
```

Additionally, download the NLTK stopwords dataset:
```python
import nltk
nltk.download('stopwords')
```

## System Requirements
- Minimum of 11 GB RAM is recommended for smooth execution.
- ![Screenshot_20231010_125826](https://github.com/kushagra8881/twitter_sentiment_analysis/assets/127012998/40260c35-60c2-478d-a735-fdd926673b32)
## positive wordcloud
![image](https://github.com/kushagra8881/twitter_sentiment_analysis/assets/127012998/243d1493-c747-4b48-9b35-c06431f9f5f5)


## Project Structure
```
twitter_sentiment_analysis/
│
├── data/                       # Directory for storing dataset
│   └── twitter_comments.csv     # Sample Twitter comments dataset
│
├── output/                     # Directory for storing output files and images
│
├── sentiment_analysis.ipynb     # Jupyter Notebook for sentiment analysis
│
└── README.md                   # Project documentation
```

## Running the Project
1. Ensure all the prerequisites are met, and the dataset (`twitter_comments.csv`) is in the `data/` directory.

2. Open the `sentiment_analysis.ipynb` notebook using Jupyter.

3. Execute each cell in the notebook sequentially to perform the following tasks:
   - Data preprocessing
   - Exploratory Data Analysis (EDA)
   - Text processing (removing stopwords, stemming, etc.)
   - Visualizations (WordCloud, Bar charts, etc.)
   - Model training (Naive Bayes, Support Vector Machine)
   - Evaluating model performance

4. The notebook will generate a visualization of positive and normal comments, saved in the `output/` directory.

5. After running the notebook, you'll be able to make predictions on new comments using the trained model.

## Note
- Ensure you have sufficient storage space for saving the output images and model files.
- The model training phase might consume a significant amount of memory, so ensure you have a minimum of 11 GB RAM available.

For any issues or inquiries, please refer to the project documentation or contact the project maintainers.

Happy analyzing!
