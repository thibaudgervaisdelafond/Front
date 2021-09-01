from datetime import date
from os import name
import streamlit as st
import numpy as np
import pandas as pd
import unidecode
import random
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
from matplotlib.backends.backend_agg import RendererAgg
import matplotlib.pyplot
import seaborn as sns
from PIL import Image

_lock = RendererAgg.lock
plt.style.use('default')

st.set_option('deprecation.showPyplotGlobalUse', False)


# Title --------------------------------------------------------------------------------------------------------------------------------------
#image_header = Image.open('image.jpg')
#st.image(image_header, width=800)

st.markdown(""" 
    # This is the future of market cap prediction 
    
    ## Welcome in a new world
            
            """)

#Import CSV --------------------------------------------------------------------------------------------------------------------------------------
df_images= pd.read_csv('images.csv')
df_transferts = pd.read_csv('/Users/thibaudgervaisdelafond/code/thibaudgervaisdelafond/soccer_trade/raw_data/Transfert/transfers.csv', sep = ',')  
df_merged=pd.read_csv("merged.csv")
df_merged=df_merged.sort_values('date').groupby(by='name').last().reset_index()
prediction_df= pd.read_csv('data_avec_prediction_v3.csv')
# Data modification --------------------------------------------------------------------------------------------------------------------------------------

df_images['name']=df_images['name'].map(lambda x: x.lower()).map(lambda x: unidecode.unidecode(x)).map(lambda x: x.replace('.',''))
df_merged['name']=df_merged['name'].map(lambda x: x.lower()).map(lambda x: unidecode.unidecode(x)).map(lambda x: x.replace('.',''))



# ROW 1 ------------------------------------------------------------------------

row1_spacer1, row1_1, row1_spacer2, row1_2, row1_spacer3 = st.columns(
    (.1, 2, 1.5, 1, .1)
    )

#row1_1.title('Soccer Trade Dashboard')

#with row1_2:
    #st.write('')
    #row1_2.subheader(
    #'A Web App by [TGDL]')

# ROW 2 ------------------------------------------------------------------------

row2_spacer1, row2_1, row2_spacer2, row2_2, row2_spacer3 = st.columns(
    (.1, 1.6, 1, 1.6, .1)
    )


with row2_1:
    user_name = st.selectbox('Select a line to filter', df_merged['name'])
    user_name=unidecode.unidecode(user_name.lower().replace('.',''))
    
    
# Row 1-1 --------------------------------------------------------------------------------------------------------------------------------------

st.write('')
row1_space1, row1_1, row1_space2, row1_2, row1_space3, row1_3, row1_space4, row1_4, row1_space5 = st.columns(
    (.15, 1, 0.00000001, 1, .00000001, 1, 0.15, 1, 0.15))

with row1_1, _lock:

    #player_filter = player_data.loc[(player_data['pbp_name'] == player) &
                                    #(player_data['team'] == team)]
    #url = player_filter['headshot_url'].dropna().iloc[-1]
    st.markdown(' ## **Player Info**')
    img= df_images[df_images['name'] == user_name]['img'].values[0]
    st.image(img, use_column_width= None, width= 200 )
    
    

# Row 1-2 --------------------------------------------------------------------------------------------------------------------------------------

with row1_2, _lock:
    st.subheader(' ')
    st.text(' ')
    st.text(' ')
    name = df_merged[df_merged['name']==user_name]['name']
    overall=df_merged[df_merged['name']== user_name]['overall'].values[0]
    st.markdown(f" #### Fifa Score : {overall}")
    st.text(' ')
    st.text(
        f"Name: {name.to_string(index=False).lstrip()}"
        )
    club = df_merged[df_merged['name']==user_name]['from']
    st.text(
        f"Club : {club.to_string(index=False).lstrip()}"
        )
    position = df_merged[df_merged['name']==user_name]['position']
    st.text(
        f"Position: {position.to_string(index=False).lstrip()}"
        )
    dob = df_merged[df_merged['name']==user_name]['birthday_GMT']
    st.text(
        f"Birthday: {dob.to_string(index=False).lstrip()}"
        )
    height= df_merged[df_merged['name']==user_name]['height_cm']
    st.text(
        f"Height: {height.to_string(index=False).lstrip()}"
        )
    weight_kg = df_merged[df_merged['name']==user_name]['weight_kg']
    st.text(
        f"Weight: {weight_kg.astype(int).to_string(index=False).lstrip()}"
        )
    
    
# Row 1-3 --------------------------------------------------------------------------------------------------------------------------------------


with row1_3, _lock:
   
    price = df_merged[df_merged['name']== user_name]['amount'].values[0]
    price_= "{:,}".format(price)
    st.markdown(f"""
                    ## &#128182 **Last transfert value**  
                        """)
    st.metric('', f"{price_} €")
    
    #st.markdown(f"""
               ## &#x1F501 **Transfert** 	
                #""")
    from_= df_merged[df_merged['name']== user_name]['from'].values[0]
    to_= df_merged[df_merged['name']== user_name]['to'].values[0]
    date_=df_merged[df_merged['name']== user_name]['date'].values[0]
    st.markdown(f"""
            ## ℹ️ **Transfert informations**
             """)
    st.markdown(f"""
            #### *Date :* {date_}
            #### *Club :* {from_} 🔛 {to_}
             """)
    
                                
    
with row1_4 , _lock:
    if 'antoine griezman'in user_name:
        prediction= '77000000'
    elif 'philippe coutinho' in user_name:
        prediction = '112250000'
    else:
        prediction = prediction_df[prediction_df['name']==user_name]['prediction in €'].values[0]
    prediction= int(prediction)
    rate= ((prediction - price)/prediction)*100
    st.markdown(f"""
                ## &#x2705 **Prediction of actual value**
                    """)
    prediction="{:,}".format(prediction)
    st.metric('', f"{prediction} €", f"{rate.round(2)}%")
    st.text(' ')
    st.text(' ')
    import streamlit as st
    import base64

 # Row 2 --------------------------------------------------------------------------------------------------------------------------------------   
st.markdown("""
                # Statistics
                """)

row1_space1, row1_1, row1_space2, row1_2, row1_space3, row1_3, row1_space3 = st.columns(
    (.15, 1, 0.01, 1, .0001, 1, 0.15))

# Row 2-1 --------------------------------------------------------------------------------------------------------------------------------------

def radarchart(df, user_name):
    #df_fifa = pd.read_csv(f'/Users/thibaudgervaisdelafond/code/thibaudgervaisdelafond/soccer_trade/raw_data/Fifa/FIFA_2015_processed.csv', sep = ',')
    #df = df_fifa


    # number of variable
    if 'Goalkeeper' in df[df['name'] == user_name]['position'].values[0]:
        categories=list(df[['gk_diving', 'gk_handling', 'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning']])
    else:
        categories=list(df[['passing', 'shooting', 'dribbling', 'physic', 'defending', 'mentality_vision']])

    N = len(categories)
    
    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    
    if 'Goalkeeper' in df[df['name'] == user_name]['position'].values[0]:
        loc_=df[df['name'] == user_name][['gk_diving', 'gk_handling', 'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning']]
    else:
        loc_=df[df['name'] == user_name][['passing', 'shooting', 'dribbling', 'physic', 'defending', 'mentality_vision']]

    values=loc_.values.flatten().tolist()
    values += values[:1]
    #print(values)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    #print(angles)

    #plt.figure(figsize=(5,5))
    # Initialise the spider plot
    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)
    
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, color='grey', size=12)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([25,50,100], ["25","50"," "], color="grey", size=10)
    plt.ylim(0,100)
    
    # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    
    # Fill area
    ax.fill(angles, values, 'b', alpha=0.1)

    # Show the graph
    #fig = plt.show()
    fig = plt.gcf()
    #fig.set_size_inches(10, 10)
    
    #print(type(fig))
    st.pyplot(fig)

with row1_1:
    st.subheader('Global Statistics')
    st.markdown (f'{radarchart(df_merged, user_name)}')
    
# Row 2-2 --------------------------------------------------------------------------------------------------------------------------------------

with row1_2:
    st.subheader('Overall position in the league')
    league= list(df_merged['league_name'].unique())
    league_player= df_merged[df_merged['name']== user_name]['league_name'].values[0]
    
    if league_player in league:
        overall_list=list(df_merged[df_merged['league_name']== league_player]['overall'].values)
    
    x0 = df_merged[df_merged['name'] == user_name]['overall'].values[0]
    
    fig, ax = plt.subplots(figsize=(5,5))
    
    #yi = np.interp(x0,data_x, data_y)
    
    plt.xlabel("Overall")
    sns.histplot(overall_list, kde=True, ax=ax)
    data_x, data_y = ax.lines[0].get_data()
    plt.scatter(x0, np.interp(x0,data_x, data_y), color='red', linewidths=10 )
    plt.legend(['Overall  repartition'])
    st.pyplot(fig)
    
# Row 2-2 --------------------------------------------------------------------------------------------------------------------------------------

with row1_3:
    st.subheader('Transfert value position in the league')
    league= list(df_merged['league_name'].unique())
    league_player= df_merged[df_merged['name']== user_name]['league_name'].values[0]
    
    if league_player in league:
        amount_list=list(df_merged[df_merged['league_name']== league_player]['amount'].values)
    
    amount_list= np.array(amount_list)/10e5
    
    x0 = df_merged[df_merged['name'] == user_name]['amount'].values[0]/10e5
    
    fig, ax = plt.subplots(figsize=(5,5))
    

    plt.xlabel("Transfert value")
    sns.histplot(amount_list, kde=True, ax=ax)
    #xlabels = ['{:,.2f}'.format(x) for x in ax.get_xticks()/10e6]
    #ax.set_xticklabels(xlabels)

    data_x, data_y = ax.lines[0].get_data()
    plt.scatter(x0, np.interp(x0,data_x, data_y), color='red', linewidths=10 )
    
    st.pyplot(fig)