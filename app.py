import dash_bootstrap_components as dbc
import dash_html_components as html
import dash
import plotly.express as px
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import dash_daq as daq
import ast
from assets.CustomerReviewGraphs import  CustomerReviewAnalysis
#from assets.CusomerReviewKeywords import CustomerReviewKeywords

def drawTrendsBarFigure(fig=None):
    if(fig is None):
        fig=resp['timeseries_bar_fig']    
    return  html.Div([dbc.Card(dbc.CardBody([html.P('Review Trends'),
    dcc.Graph(figure=fig.update_layout(
    template='plotly_dark',plot_bgcolor= 'rgba(0, 0, 0, 0)',paper_bgcolor= 'rgba(0, 0, 0, 0)',),config={
    'displayModeBar': False})])),])

def drawRatings(fig=None):
    if(fig is None):
        fig=resp['rating_fig']
    return  html.Div([dbc.Card(dbc.CardBody([html.P('Ratings Visualization'),
    dcc.Graph(figure=fig.update_layout(template='plotly_dark',plot_bgcolor= 'rgba(0, 0, 0, 0)',
    paper_bgcolor= 'rgba(0, 0, 0, 0)',),
    config={'displayModeBar': False
    })])), ])


def drawTimeSeriesFigure(fig=None):
    if(fig is None):
        fig=resp['timeseries_line_fig']
    return  html.Div([dbc.Card(dbc.CardBody([html.P('Review Sentiments Trend'),dcc.Graph(figure=fig.update_layout(template='plotly_dark',
    plot_bgcolor= 'rgba(0, 0, 0, 0)',paper_bgcolor= 'rgba(0, 0, 0, 0)',),
    config={'displayModeBar': False})])), ])

def drawPieFigure(fig=None):
    if(fig is None):
        fig=resp['sentiment_fig']
    return  html.Div([dbc.Card(dbc.CardBody([html.P('Review Sentiments'),dcc.Graph(
    figure=fig.update_layout(template='plotly_dark',
    plot_bgcolor= 'rgba(0, 0, 0, 0)',paper_bgcolor= 'rgba(0, 0, 0, 0)',),
    config={'displayModeBar': False}) ])),])
def map_sentiment(rating):
    if(int(rating)==3):
        return 2
    elif(int(rating)<3):
        return 3
    else:
        return 1 
def drawTrendingWords(pos_keywords=None,neg_kewords=None):
    return  html.Div([
    dbc.Card(
    dbc.CardBody([html.P('Trending Keywords'),dbc.ListGroup([
    dbc.ListGroupItem(pos_keywords[0], color="success"),
    dbc.ListGroupItem(neg_kewords[0], color="danger"),
    dbc.ListGroupItem(pos_keywords[1], color="success"),
    dbc.ListGroupItem(neg_kewords[1], color="danger"),
    dbc.ListGroupItem(pos_keywords[2], color="danger"),
    dbc.ListGroupItem(neg_kewords[2], color="danger"),
    dbc.ListGroupItem(pos_keywords[3], color="success"),
    dbc.ListGroupItem(neg_kewords[3], color="danger"),
    dbc.ListGroupItem(neg_kewords[4], color="danger")
    ])])),], style={'textAlign': 'center'})

def drawRecentReviews(data):
    return  html.Div([
    dbc.Card(
    dbc.CardBody([html.P('Recent Reviews'),
    dbc.Table.from_dataframe(data, striped=False, bordered=False, hover=True,dark=True,responsive=True)
    ])),], style={'textAlign': 'center',"maxHeight": "500px", "overflow": "scroll"})


review_business_data_merged = pd.read_csv('data/yelp_reviews_business_merged.csv')          
review_sentiments=[map_sentiment(s) for s in review_business_data_merged['stars_x']]
review_business_data_merged['sentiments']=review_sentiments
review_business_data_merged['date']=pd.to_datetime(review_business_data_merged.date)
data_most_reviewed_store=review_business_data_merged[review_business_data_merged['business_id']=='4CxF8c3MB7VAdY8zFb2cZQ'].sort_values(by=['date'])
date_str=[en.strftime('%Y') for  en in data_most_reviewed_store['date']]
data_most_reviewed_store['date']=date_str
data_most_reviewed_store_timeseries_pos=data_most_reviewed_store[data_most_reviewed_store['sentiments']==1].groupby('date')['date','sentiments'].sum()
data_most_reviewed_store_timeseries_neg=data_most_reviewed_store[data_most_reviewed_store['sentiments']==3].groupby('date')['date','sentiments'].sum()
data_most_reviewed_store_timeseries_neu=data_most_reviewed_store[data_most_reviewed_store['sentiments']==2].groupby('date')['date','sentiments'].sum()
data_most_reviewed_store_timeseries_pos['rating']=['Positive' for i in range(len(data_most_reviewed_store_timeseries_pos))]
data_most_reviewed_store_timeseries_neg['rating']=['Negative' for i in range(len(data_most_reviewed_store_timeseries_neg))]
data_most_reviewed_store_timeseries_neu['rating']=['Neutral' for i in range(len(data_most_reviewed_store_timeseries_neu))]
data_most_reviewed_store_timeseries=pd.concat([data_most_reviewed_store_timeseries_pos,data_most_reviewed_store_timeseries_neg])
data_most_reviewed_store_timeseries=pd.concat([data_most_reviewed_store_timeseries,data_most_reviewed_store_timeseries_neu])



# Text field
def drawText(text='NA'):
    return html.Div([
    dbc.Card(
    dbc.CardBody([html.Div([html.P('Selected Business',style={'color': 'white'}),html.H5(text,style={'color': 'yellow'}), ], style={'textAlign': 'center'}) 
    ])),])

def drawTextOverallRating(text='NA'):
    return html.Div([
    dbc.Card(dbc.CardBody([html.Div([html.P('Overall Rating',style={'color': 'white'}),html.H5(text,style={'color': 'yellow'}), ], style={'textAlign': 'center'}) 
    ])),])

def drawTotalReview(text='Text'):
    return html.Div([
        dbc.Card(
        dbc.CardBody([html.Div([html.P('Total Review',style={'color': 'white'}), 
        html.H5(text,style={'color': 'yellow'}), 
        ], style={'textAlign': 'center'}) ])),])
def drawBusinessName(): 
    return html.Div([
    dbc.CardImg(src="assets/RAAS.png", top=True,style={'height':'62px','width':'400px','padding':'1px'}),    
    dcc.Dropdown(id='business_name',
    options=[
    {'label': 'Voodoo Doughnut - Old Town', 'value': '4CxF8c3MB7VAdY8zFb2cZQ'},
    {'label': 'Screen Door', 'value': 'OQ2oHkcWA8KNC1Lsvj1SBA'},
    {'label': 'Mike Pastry', 'value': 'PrsvO1rzkgg6qFizlAoEtg'},
    {'label': 'Neptune Oyster', 'value': 'y2w6rFaO0XEiG5mFfOsiFA'}
    ],
    placeholder="Select business",)])    

keywords=pd.read_csv('data/trending_keywords.csv')
def get_Keywords(keywords,id):
    keywords=keywords[keywords['id']==id].iloc[0]
    pos = ast.literal_eval(keywords['pos_keywords'])
    neg = ast.literal_eval(keywords['neg_keywords'])
    return pos,neg

def get_recent_reviews(review_business_data_merged,id):
    data_most_reviewed_store=review_business_data_merged[review_business_data_merged['business_id']==id].sort_values(by=['date'],ascending=False)
    sentimments_dict={3:'Negative',2:'Neutral',1:'Positive'}
    sentiment_names=[sentimments_dict[int(i)] for i in data_most_reviewed_store['sentiments'].values]
    data_most_reviewed_store['sentiment']=sentiment_names
    data_most_reviewed_store=data_most_reviewed_store.drop(columns=['business_id','review_id','user_id','user_id','categories'])
    data_most_reviewed_store=data_most_reviewed_store[['text','sentiment','date','stars_x']]
    data_most_reviewed_store=data_most_reviewed_store.rename(columns={'stars_x':'rating','text':'review'})
    return data_most_reviewed_store

# Build App
app = dash.Dash(external_stylesheets=[dbc.themes.SLATE])
cr=CustomerReviewAnalysis()
resp=cr.getReviewAnalysis(review_business_data_merged,'4CxF8c3MB7VAdY8zFb2cZQ')
recent_reviews=get_recent_reviews(review_business_data_merged,'4CxF8c3MB7VAdY8zFb2cZQ')
#ck=CustomerReviewKeywords()
#ck.get_trending_keywords(data_most_reviewed_store,num_keywords=10)
pos_keywords,neg_keywords=get_Keywords(keywords,'4CxF8c3MB7VAdY8zFb2cZQ')
app.layout = html.Div([dbc.Card(
dbc.CardBody([dbc.Row([dbc.Col([drawBusinessName()], width=3),dbc.Col([drawText()], width=3,id='sb01'),dbc.Col([drawTextOverallRating()], width=3,id='sb02'),dbc.Col([drawTotalReview()], width=3,id='sb03'),], align='center'), html.Br(),
dbc.Row([dbc.Col([drawPieFigure()], width=3,id='df02'),dbc.Col([drawTrendingWords(pos_keywords,neg_keywords)], width=3,id='sb04'),dbc.Col([drawRecentReviews(recent_reviews)], width=6,id='df04'),], align='center'), html.Br(),
dbc.Row([dbc.Col([drawTrendsBarFigure()], width=5,id='df03'),dbc.Col([drawTimeSeriesFigure()], width=4,id='df05'),dbc.Col([drawRatings()],width=3,id='df01')], align='center'),]), color = 'dark')])

@app.callback(
    [Output(component_id='sb01', component_property='children'),
    Output(component_id='sb02', component_property='children'),Output(component_id='sb03', component_property='children'),
    Output(component_id='df01', component_property='children'),Output(component_id='df02', component_property='children'),
    Output(component_id='df04', component_property='children'),Output(component_id='df03', component_property='children'),
    Output(component_id='sb04', component_property='children')],
    [Input(component_id='business_name', component_property='value')]
)
def update_output_div(input_value):
    cr=CustomerReviewAnalysis()
    resp=cr.getReviewAnalysis(review_business_data_merged,input_value)
    pos_keywords,neg_keywords=get_Keywords(keywords,input_value)
    recent_reviews=get_recent_reviews(review_business_data_merged,input_value)
    return drawText(resp['business_name']),drawTextOverallRating(resp['business_rating']),drawTotalReview(resp['num_review']),drawRatings(resp['rating_fig']),drawPieFigure(resp['sentiment_fig']),drawRecentReviews(recent_reviews),drawTrendsBarFigure(resp['timeseries_bar_fig']),drawTrendingWords(pos_keywords,neg_keywords)

if __name__ == '__main__':
    app.run_server(debug=False)