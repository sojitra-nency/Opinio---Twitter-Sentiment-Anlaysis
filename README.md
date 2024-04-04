# Opinio: Sentiment Analysis for Tweets
Opinio is a web application that helps you understand the overall sentiment of a tweet, classifying it as positive or negative.

## Dataset

The training data used in this project is not included in the repository due to its size. However, you can download the dataset from Kaggle at the following link:

#### Sentiment140: 
https://www.kaggle.com/datasets/kazanova/sentiment140

This dataset contains a large collection of tweets labeled with their sentiment (positive, neutral, or negative). You can use this data to train and evaluate your own sentiment analysis model.

## Functionality
Users enter a tweet in the provided text area.

The app processes the tweet using a pre-trained logistic regression model with 81% training accuracy and 77% testing accuracy.

The processed tweet is fed into the model, providing a sentiment classification (positive or negative).

The predicted sentiment is displayed on the screen, potentially with visual cues (optional).

## Technologies Used
### Backend:
Python

scikit-learn (Logistic Regression)

Flask (optional)

Streamlit (optional)

Pickle (for model and vectorizer serialization)

### Frontend:
HTML (optional, for Flask)

## Optional Features:
Streamlit for a user-friendly interface (alternative to Flask)

CSS for styling the user interface

## Project Structure
Opinio/
├── app.py (optional, Flask implementation)

├── main.py (backend with logistic regression model)

├── README.md (this file)

├── requirements.txt (dependencies)

├── streamlit.py (optional, Streamlit implementation)

└── templates/ (optional, HTML templates for Flask)

    └── index.html
    
    └── result.html
    
├── trained_model.sav (saved logistic regression model)

└── vectorizer.pickle (saved vectorizer)


#### Note: 
You can choose to implement the backend using either Flask or Streamlit. Flask provides more flexibility for building a custom web interface, while Streamlit offers a quicker way to deploy a simple interface.

## Installation and Usage
### Prerequisites:

Python 3.x
Install required libraries using pip install -r requirements.txt

#### Flask Implementation (optional):

Run the Flask app: python app.py

Access the app in your web browser (usually at http://127.0.0.1:5000/)

#### Streamlit Implementation (optional):

Run the Streamlit app: streamlit run streamlit.py

Access the app in your web browser (usually at http://localhost:8501/)

#### Note:

The specific port used by Flask or Streamlit might differ. Check the console output for the exact URL.

## Contributing
We welcome contributions to improve Opinio. Feel free to fork the repository, make changes, and submit a pull request.
