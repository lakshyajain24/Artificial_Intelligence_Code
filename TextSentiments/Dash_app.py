#Importing Dash Libraries
import dash
import dash_html_components as html 
from dash.dependencies import Input,Output,State
import dash_core_components as dcc
import webbrowser
from TextSentimentsPrediction import * #Calling functions from TextSentimentsPrediction.py

#Open Browser
def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")

#Preprocesing the Database
def globalization():
    global df
    df=pd.read_csv('balanced_reviews.csv')
    df=df[df["overall"]!=3]
    j=0
    global webdf
    webdf={}
    df2=df.iloc[:,:-1].sample(10)
    for i in range(0,len(df2)):
        while j<len(df2):
            webdf[df2.iloc[i,1]]=df2.iloc[j,0]
            j+=1
            break


#Dash App
app = dash.Dash()

@app.callback(
    Output("output_button","value"),
    [Input("input","value")]
)
  
def update_value(input_data):
    try:
        sent = sentiment_predict(Prepare_text(input_data))
        output = Conditions_Check(sent)
        return output
    except:
        return "Error"

@app.callback(
    Output("output","children"),
    [
    Input("output_button","value"),
    Input("output_button","n_clicks")
    ]
    )    


def update_button_review(value,clicks):
    if clicks>0 and value: 
        if value=="Positive":
            return value
        elif value=="Negative":
            return value
        
@app.callback(
    Output("output_button","n_clicks"),
    [
     Input("output_button","value")
     ]
    )
def n_click(value):
    if not value:
        return 0

#loading all the files
def load_model():
    #loading pickle and csv file in memory
    print("Busy in loading the model in memory")
    global df
    df=pd.read_csv("balanced_reviews.csv")
    file=open("pickle_model.pkl","rb")
    global pickle_model
    pickle_model=pickle.load(file)
    global vocab
    file=open("feature.pkl","rb")
    vocab=pickle.load(file)

#Defining the main
def main():
    load_model() 
    globalization()
    open_browser()
    
if __name__=="__main__":
#calling the main function
    main()

#Building the Webpage
    app.layout = html.Div(children =[
    html.H1(id="mainheading", children="Text Sentiment Review"),
    dcc.Textarea(id ='input',placeholder="Enter Text>>",style={"width":"50%","height":"100px","text-align":"center"}),
        html.Br(children=None), html.Br(children=None),
    dcc.Dropdown(id="review", options=[{"label":i,"value":j}for i,j in webdf.items()],placeholder="Select Review"),
        html.Br(children=None),
    html.Button(id="output_button",children="Find",style={"align":"center","height":"40px", "width":"20%"},n_clicks=0),
        html.Br(children=None),
    html.H2(id="output",children=None,style={"color":"red","align":"center","font-weight":"bold","font-style":"oblique"})
    ],style={"text-align":"center","background-color":"#EBEBEB"})
    app.run_server()