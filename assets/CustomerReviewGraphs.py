# Import necessary libraries
import pandas as pd
import plotly as py
import cufflinks
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
from branca.element import Figure
import folium 
import matplotlib.pyplot as pPlot
import numpy as npy
from PIL import Image
from IPython.display import Image as img

class CustomerReviewAnalysis:
    def getStarRatings(self,data_most_reviewed_store):   
        rating_5=len(data_most_reviewed_store[data_most_reviewed_store['stars_x']==5])
        rating_4=len(data_most_reviewed_store[data_most_reviewed_store['stars_x']==4])
        rating_3=len(data_most_reviewed_store[data_most_reviewed_store['stars_x']==3])
        rating_2=len(data_most_reviewed_store[data_most_reviewed_store['stars_x']==2])
        rating_1=len(data_most_reviewed_store[data_most_reviewed_store['stars_x']==1])          
        fig = go.Figure()
        fig.add_trace(go.Bar(y=[1],x=[rating_1],name='1',orientation='h',marker=dict(color='rgb(255, 51, 51)')))
        fig.add_trace(go.Bar(y=[2],name='2',x=[rating_2],orientation='h',marker=dict(
        color='rgb(255, 92, 51)')))
        fig.add_trace(go.Bar(y=[3],name='3',x=[rating_3],orientation='h',marker=dict(color='rgb(255, 255, 77)')))
        fig.add_trace(go.Bar(y=[4],name='4',x=[rating_4],orientation='h',marker=dict(color='rgb(77, 255, 166)')))
        fig.add_trace(go.Bar(y=[5],name='5',x=[rating_5],orientation='h',marker=dict(color='rgb(166, 255, 77)')))
        return fig
    
    def getTimeSeries(self,data_most_reviewed_store):
        date_str=[en.strftime('%Y') for  en in data_most_reviewed_store['date']]
        data_most_reviewed_store['date_year']=date_str
        # Sum the number of reviews per year
        data_most_reviewed_store_timeseries_pos=data_most_reviewed_store[data_most_reviewed_store['sentiments']==1].groupby('date_year')['date_year','sentiments'].sum()
        data_most_reviewed_store_timeseries_neg=data_most_reviewed_store[data_most_reviewed_store['sentiments']==3].groupby('date_year')['date_year','sentiments'].sum()
        data_most_reviewed_store_timeseries_neu=data_most_reviewed_store[data_most_reviewed_store['sentiments']==2].groupby('date_year')['date_year','sentiments'].sum()
        data_most_reviewed_store_timeseries_pos['rating']=['Positive' for i in range(len(data_most_reviewed_store_timeseries_pos))]
        data_most_reviewed_store_timeseries_neg['rating']=['Negative' for i in range(len(data_most_reviewed_store_timeseries_neg))]
        data_most_reviewed_store_timeseries_neu['rating']=['Neutral' for i in range(len(data_most_reviewed_store_timeseries_neu))]
        data_most_reviewed_store_timeseries=pd.concat([data_most_reviewed_store_timeseries_pos,data_most_reviewed_store_timeseries_neg])
        data_most_reviewed_store_timeseries=pd.concat([data_most_reviewed_store_timeseries,data_most_reviewed_store_timeseries_neu])
        # No of reviews per year
        data_most_reviewed_store_timeseries['year']=data_most_reviewed_store_timeseries.index
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(data_most_reviewed_store_timeseries.index), 
                         y=data_most_reviewed_store_timeseries_pos['sentiments'],
                         mode='lines',
                         name='positive',
                         line=dict(color='rgb(0,245,153)', width=1)))
        fig.add_trace(go.Scatter(x=list(data_most_reviewed_store_timeseries.index), 
                         y=data_most_reviewed_store_timeseries_neg['sentiments'],
                         mode='lines',
                         name='negative',
                         line=dict(color='rgb(255, 102, 102)', width=1)))
        fig.add_trace(go.Scatter(x=list(data_most_reviewed_store_timeseries.index), 
                         y=data_most_reviewed_store_timeseries_neu['sentiments'],
                         mode='lines',
                         name='neutral',
                         line=dict(color='rgb(102, 102, 255)', width=1)))
        timeseries_line_fig=fig
        timeseries_bar_fig = px.bar(data_most_reviewed_store_timeseries, x='year', y='sentiments',color='rating')
        return timeseries_line_fig,timeseries_bar_fig



    def getReviewAnalysis(self,review_business_data_merged,store_id):  
        data_most_reviewed_store=review_business_data_merged[review_business_data_merged['business_id']==store_id].sort_values(by=['date'])  
        business_name=data_most_reviewed_store.iloc[0]['name']
        business_lat=data_most_reviewed_store.iloc[0]['latitude']
        business_long=data_most_reviewed_store.iloc[0]['longitude']
        business_rating=data_most_reviewed_store.iloc[0]['stars_y']
        num_review=len(data_most_reviewed_store)
        rating_fig=self.getStarRatings(data_most_reviewed_store)
        num_pos_review=str(len(data_most_reviewed_store[data_most_reviewed_store['sentiments']==1]))
        num_neu_review=str(len(data_most_reviewed_store[data_most_reviewed_store['sentiments']==2]))
        num_neg_review=str(len(data_most_reviewed_store[data_most_reviewed_store['sentiments']==3]))
        sentimments_dict={3:'Negative',2:'Neutral',1:'Positive'}
        sentiment_names=[sentimments_dict[int(i)] for i in data_most_reviewed_store['sentiments'].values]
        data_most_reviewed_store['sentiment_name']=sentiment_names
        sentiment_fig = px.pie(data_most_reviewed_store, values='sentiments', names='sentiment_name',color='sentiment_name',color_discrete_map={'Neutral':'yellow','Negative':'cyan','Positive':'green'})
        timeseries_line_fig,timeseries_bar_fig=self.getTimeSeries(data_most_reviewed_store)
        output={}
        output['business_name']=business_name
        output['num_review']=num_review
        output['business_lat']=business_lat
        output['business_rating']=business_rating
        output['business_long']=business_long
        output['rating_fig']=rating_fig
        output['sentiment_fig']=sentiment_fig
        output['timeseries_line_fig']=timeseries_line_fig
        output['timeseries_bar_fig']=timeseries_bar_fig
        return output