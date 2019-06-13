import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from sklearn.externals import joblib
from sqlalchemy import create_engine
from plotly.graph_objs import Bar


app = Flask(__name__)


def tokenize(text):
    """
    method to tokenize text
    :param text: raw text
    :return: tokenized text
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages_w_categories', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # pre work for visuals
    graphs = []

    categories = df.iloc[:, 4:].sum().sort_values(ascending=False)
    top_cats = categories.index[:10]

    genres = df.groupby('genre').sum()[top_cats]
    data = []
    for cat in genres.columns[1:]:
        data.append(Bar(x=genres.index, y=genres[cat], name=cat))

    # create visuals
    g1 = {'data': [Bar(x=categories.index, y=categories, marker=dict(color='Blue'), opacity=0.75)],
          'layout': {
                'title': 'Messages per Category',
                'yaxis': {'title': "Message Count"},
                'xaxis': {'title': "Categories"}}}
    graphs. append(g1)

    for idx, cat in enumerate(genres.columns[1:]):
        title = 'Categories per ' + cat
        g = {'data': [data[idx]],
             'layout': {
                'title': title,
                'yaxis': {'title': "Messages per Category"},
                'xaxis': {'title': "Genres"}}}
        graphs.append(g)

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
