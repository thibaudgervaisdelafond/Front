

import streamlit as st

import numpy as np
import pandas as pd
import unidecode
import random
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
from math import pi

st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown(""" 
    # This is the future of market cap prediction 
    
    ## Welcome in a new world
            
            """)
df_images= pd.read_csv('images.csv')
df_transferts = pd.read_csv('/Users/thibaudgervaisdelafond/code/thibaudgervaisdelafond/soccer_trade/raw_data/Transfert/transfers.csv', sep = ',')  
df_merged=pd.read_csv("merged.csv")

#user_input = st.sidebar.selectbox('Select a line to filter', df_merged['name'])

df_images['name']=df_images['name'].map(lambda x: x.lower()).map(lambda x: unidecode.unidecode(x)).map(lambda x: x.replace('.',''))
df_merged['name']=df_merged['name'].map(lambda x: x.lower()).map(lambda x: unidecode.unidecode(x)).map(lambda x: x.replace('.',''))

#---------------------------------------------------------------------------------------------------------#
#------------------------------------------RadarChart-----------------------------------------------------#
#---------------------------------------------------------------------------------------------------------#
# Set data

df=pd.read_csv("merged.csv")
    
def radarchart(df, user_name):
    #df_fifa = pd.read_csv(f'/Users/thibaudgervaisdelafond/code/thibaudgervaisdelafond/soccer_trade/raw_data/Fifa/FIFA_2015_processed.csv', sep = ',')
    #df = df_fifa


    # number of variable
    categories=list(df[['passing', 'shooting', 'dribbling', 'pace', 'defending', 'physic']])

    N = len(categories)
    
    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:

    loc_=df[df['name'] == user_name][['passing', 'shooting', 'dribbling', 'pace', 'defending', 'physic']]

    values=loc_.values.flatten().tolist()
    values += values[:1]
    #print(values)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    #print(angles)

    plt.figure(figsize=(5,5))
    # Initialise the spider plot
    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)
    
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([25,50,100], ["25","50"," "], color="grey", size=7)
    plt.ylim(0,100)
    
    # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    
    # Fill area
    ax.fill(angles, values, 'b', alpha=0.1)

    # Show the graph
    fig = plt.show()
    #print(type(fig))
    st.pyplot(fig)


user_name = st.selectbox('Select a line to filter', df_merged['name'])
user_name=unidecode.unidecode(user_name.lower().replace('.',''))

if user_name:
    row1_spacer1, row1_1, row1_spacer2, row1_2, row1_spacer3 = st.columns(
    (.1, 2, 1.5, 1, .1)
    )

    col1, col2, col3= st.columns(3)
    if st.sidebar.checkbox('image'):
        img=df_images[df_images['name'] == user_name]['img'].values[0]
        col1.header("Player")
        col1.image(img, use_column_width=None)
    
    if st.sidebar.checkbox('Club'):
        col3.header("Club")
        club=df_merged[df_merged['name']== user_name]['club_name'].values[0]
        col3.markdown(f"""
                    # {club}
                    """)
    
    if st.sidebar.checkbox('Position'):
        col2.header('Position')
        position=df_merged[df_merged['name']== user_name]['position'].values[0]
        col2.markdown(f"""
                    # {position}
                    #""")
        if position=='Midfielder':
            col2.header(f'Rank in league to {position}')
            rank=df_merged[df_merged['name']== user_name]['rank_in_league_top_midfielders'].values[0]
            col2.markdown(f"""
                    # {rank}
                    #""")
    
    #col4.header("Fifa")
    #overall=df_merged[df_merged['name']== user_name]['overall'].values[0]
    #col4.markdown(f"""
                 # {overall}
                 # """)
    
    #price = df_merged[df_merged['name']== user_name]['amount'].values[0]
    #col3.header("Market cap")
    #col3.markdown(f"""
                # {price} &#129297
                 #""")
    
    
    
    #col3.image(radarchart(df_merged, user_name), use_column_width=None)
    #radarchart(df_merged, user_name)

if user_name:
    
    col1, col2 = st.columns(2)
    if st.sidebar.checkbox('Fifa score'):
        col1.header("Fifa")
        overall=df_merged[df_merged['name']== user_name]['overall'].values[0]
        col1.markdown(f"""
                    # {overall}
                    """)
    if st.sidebar.checkbox('Performances'):
        with col2:
            col2.header('Performances')
            radarchart(df_merged, user_name)


if st.sidebar.checkbox('Marketcap'):
    price = df_merged[df_merged['name']== user_name]['amount'].values[0]
    st.markdown(f"""
                # Marketcap
                ## {price} &#129297
                    """)