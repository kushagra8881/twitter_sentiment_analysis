from flask import Flask, render_template, request, send_file
from docx import Document
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import joblib
import os

# Load the model
predictor = joblib.load('/home/kushagra/Documents/nlp/project/twitter_sentiment_analysis/my_model.pkl')

# Load the training data and preprocess it
train_data = pd.read_csv('/home/kushagra/Documents/nlp/project/twitter_sentiment_analysis/train_av.csv')
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
cv = CountVectorizer()

def preprocess_tweet(tweet):
    review = re.sub('[^a-zA-Z]', ' ', tweet)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    return ' '.join(review)

train_data['changed_tweet'] = train_data['tweet'].apply(preprocess_tweet)
train_data_vectorize = cv.fit_transform(train_data['changed_tweet']).toarray()

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        input_df = pd.read_csv(uploaded_file)
        input_df['changed_tweet'] = input_df['tweet'].apply(preprocess_tweet)
        content_answer = cv.transform(input_df['changed_tweet']).toarray()
        content_answer = predictor.predict(content_answer)
        input_df['predict'] = content_answer
        output_file = save_csv(input_df)
        return send_file(output_file, as_attachment=True, download_name='output.csv', mimetype='text/csv')
    else:
        return "No file uploaded."

def save_csv(output_df):
    output_dir = '/home/kushagra/Documents/nlp/project/twitter_sentiment_analysis/'
    output_file = os.path.join(output_dir, 'output.csv')
    output_df.to_csv(output_file, index=False)
    return output_file

if __name__ == '__main__':
    app.run(debug=True)

