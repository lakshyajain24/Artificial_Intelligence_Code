# Importing the libraries
import pickle
import pandas as pd
import webbrowser
# !pip install dash
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from matplotlib import pyplot as plt

from dash.dependencies import Input, Output , State
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import os
#import wordcloud
from collections import Counter
import numpy as np
#from wordcloud import WordCloud, STOPWORDS

# Declaring Global variables
project_name = None
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# Defining My Functions
def load_model():
    global scrappedReviews
    scrappedReviews = pd.read_csv('scrappedReviews.csv')
  
    global pickle_model
    file = open("pickle_model.pkl", 'rb') 
    pickle_model = pickle.load(file)

    global vocab
    file = open("feature.pkl", 'rb') 
    vocab = pickle.load(file)
    #pie chart
    print('Loading Data......')

    #drop down
    global chart_dropdown_values
    chart_dropdown_values = {}
    for i in range(400,501):
        chart_dropdown_values[scrappedReviews['reviews'][i]] = scrappedReviews['reviews'][i]
    chart_dropdown_values = [{"label":key, "value":values} for key,values in chart_dropdown_values.items()]
    
def check_review(reviewText):

    #reviewText has to be vectorised, that vectorizer is not saved yet
    #load the vectorize and call transform and then pass that to model preidctor
    #load it later

    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)
    vectorised_review = transformer.fit_transform(loaded_vec.fit_transform([reviewText]))


    # Add code to test the sentiment of using both the model
    # 0 == negative   1 == positive
    
    return pickle_model.predict(vectorised_review)

def open_browser():
    webbrowser.open_new('http://127.0.0.1:8050/')
    
def create_app_ui():
    main_layout = html.Div(
    [
    html.H1(id='Main_title', children = "Sentiment Analysis with Insights",style={'text-align':'center'}),
    html.Hr(style={'background-color':'black'}),
    html.H2(children = "Pie Chart",style = {'text-align':'center','text-decoration':'underline'}),
    html.P([html.Img(src=app.get_asset_url('sentiment.png'),style={'width':'700px','height':'400px'})],style={'text-align':'center'}),
    html.Hr(style={'background-color':'black'}),
    html.H2(children = "WordCloud",style = {'text-align':'center','text-decoration':'underline'}),
    html.P([html.Img(src=app.get_asset_url('wordCloud.png'),style={'width':'700px','height':'400px'})],style={'text-align':'center'}),
    html.Hr(style={'background-color':'black'}),
    html.H2(children = "Select a Review",style = {'text-align':'center','text-decoration':'underline'}),
    dcc.Dropdown(
                id='Chart_Dropdown', 
                  options=chart_dropdown_values,
                  placeholder = 'Select a Review',style={'font-size':'22px','height':'70px'}
                    ),
    html.H1(children = 'Missing',id='sentiment1',style={'text-align':'center'}),
    html.Hr(style={'background-color':'black'}),
    html.H2(children = "Find Sentiment of Your Review",style = {'text-align':'center','text-decoration':'underline'}),
    dcc.Textarea(
        id = 'textarea_review',
        placeholder = 'Enter the review here.....',
        style = {'width':'100%', 'height':150,'font-size':'22px'}
        ),
    
    dbc.Button(
        children = 'FInd Review',
        id = 'button_review',
        color = 'dark',
        style= {'width':'100%'}
        ),
    
    html.H1(children = 'Missing', id='result',style={'text-align':'center'})
    
    ]    
    )
    
    return main_layout



'''
Event Handling 
When some clicks the button call my method update_app_ui

Wiring 
Object      Event    Function 
Button      Click    update_app_ui

Decorators and callbacks mechanism is a way to implment wiring in python
Input  === Arguments to your callback
Output === return of your callback 

'''

'''
@app.callback(
    Output( 'result'   , 'children'     ),
    [
    Input( 'textarea_review'    ,  'value'    )
    ]
    )
def update_app_ui(textarea_value):
    
    print("Data Type = ", str(type(textarea_value)))
    print("Value = ", str(textarea_value))

    response = check_review(textarea_value)

    if (response[0] == 0):
        result = 'Negative'
    elif (response[0] == 1 ):
        result = 'Positive'
    else:
        result = 'Unknown'

    return result
'''


@app.callback(
    Output( 'result'   , 'children'     ),
    [
    Input( 'button_review'    ,  'n_clicks')
    ],
    [
    State( 'textarea_review'  ,   'value'  )
    ]
    )
def update_app_ui_2(n_clicks, textarea_value):

    print("Data Type = ", str(type(n_clicks)))
    print("Value = ", str(n_clicks))


    print("Data Type = ", str(type(textarea_value)))
    print("Value = ", str(textarea_value))


    if (n_clicks > 0):

        response = check_review(textarea_value)
        if (response[0] == 0):
            result = 'Negative'
        elif (response[0] == 1 ):
            result = 'Positive'
        else:
            result = 'Unknown'
        
        return result
        
    else:
        return ""

@app.callback(
    Output("sentiment1", "children"),
    [Input("Chart_Dropdown", "value")])
def update_sentiment(review1):
    sentiment = []
    if review1:
        if check_review(review1)==0:
            sentiment='Negative' 
        if check_review(review1)==1:
            sentiment='Positive'
    else:
        sentiment='Missing'
    return sentiment
# Main Function to control the Flow of your Project
def main():
    print("Start of your project")
    load_model()
    #open_browser()
    #update_app_ui()
    
    
    global scrappedReviews
    global project_name
    global app
    
    project_name = "Sentiment Analysis with Insights"
    #print("My project name = ", project_name)
    #print('my scrapped data = ', scrappedReviews.sample(5) )
    
    # favicon  == 16x16 icon ----> favicon.ico  ----> assests
    app.title = project_name
    app.layout = create_app_ui()
    app.run_server()
    
    
    
    print("End of my project")
    project_name = None
    scrappedReviews = None
    app = None
    
        
# Calling the main function 
if __name__ == '__main__':
    main()
