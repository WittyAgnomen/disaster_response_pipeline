import sys
import re
import pandas as pd
import pickle
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath):
    """
    method to load database into data frame
    :param database_filepath:
    :return: X input, y output, and category names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("messages_w_categories", con=engine)
    X = df["message"]
    y = df.drop(['message', 'genre', 'id', 'original'], axis = 1)
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """
    method to prepare data for model
    :param text: raw test
    :return: text tokenized
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())  # clean non alpaha and numeric and lower
    unprepared_tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [token for token in unprepared_tokens if token not in stop_words]  # drop stop words
    tokens = [lemmatizer.lemmatize(token).strip() for token in tokens]  # lemmatize and strip
    return tokens


def build_model():
    """
    method to build and optimize model (3 fold cross validation over grid search parameters)
    :return: optimized model
    """
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    parameters = { #'vect__ngram_range': ((1, 1), (1, 2)),
              'clf__estimator__n_estimators': [20, 80],
              'clf__estimator__min_samples_split': [2, 4]} 
    model = GridSearchCV(pipeline, param_grid=parameters, verbose=4) 
    return model
    
    
def evaluate_model(model, X_test, y_test):
    """
    method to evaluate model results
    :param model:
    :param X_test: testing inputs
    :param y_test: testing outputs
    :return: prints classification report for each category
    """
    y_predicted = model.predict(X_test)
    for i, column in enumerate(y_test):
            print(column + ': =======================================')
            print(classification_report(y_test[column], y_predicted[:, i]))
            print('==================================================')


def save_model(model, model_filepath):
    """
    method to save model to pickle
    :param model:
    :param model_filepath:
    :return:
    """
    with open(model_filepath, 'wb') as outfile:
        pickle.dump(model, outfile)

        
def main():
    """
    main method to load, build, train, evaluate and save model
    :return:
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
