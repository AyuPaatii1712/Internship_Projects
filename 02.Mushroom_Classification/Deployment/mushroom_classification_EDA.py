# Import the Libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
#import numpy as np


import pandas_profiling as pp
from streamlit_pandas_profiling import st_profile_report

import sweetviz as sv
import codecs
import streamlit.components.v1 as components


st.set_page_config(page_title='Mushroom Classification', page_icon='mushroom1.jpg')
st.image('mushroom31.jpg')
st.header('Mushroom Classification')
st.write('---')


# Input DataSet Used to Train the Model---------------------------------------------------------------------------------------------------------------------------------------

st.sidebar.header('User Input Parameters')
st.sidebar.write('---')
data_file = st.sidebar.selectbox(
    label ="Upload the data or go with the Default data",
    options=['Default','Upload'])


# To use default file for training the model
if data_file == 'Default':
    st.subheader('Input DataFrame')
    data = pd.read_csv('mushrooms.csv')
    st.dataframe(data)
    use_cols = ['class','spore-print-color','gill-color','gill-size','stalk-root',
                'habitat','stalk-shape','odor','population']
    data_label = data[use_cols]

# Upload another file
if data_file == 'Upload':
    file = st.sidebar.file_uploader('Upload DataSet In "csv" formate', type = 'csv')
    
    if file == None:
        st.error('Please Upload the file')
        st.stop()
        
    else:
        data = pd.read_csv(file)
        data1 = pd.read_csv('mushrooms.csv')
        st.subheader('Input DataSet')
        st.dataframe(data)
        
        # Columns use to Train the Model (columns which are more important, based on Feature Importance)
        use_cols = ['class','spore-print-color','gill-color','gill-size','stalk-root',
                    'habitat','stalk-shape','odor','population']
        
        # What to do if Columns is Present OR Not Present
        try:
            data_label = data[use_cols]

        except:
            st.error('Please Upload the correct file, your file must contain below columns')
            st.write(pd.DataFrame(use_cols, columns=['columns']))
            st.stop()

# Input DataSet is Taken------------------------------------------------------------------------------------------------------------------------------------------------------


if 'eda_report' not in st.session_state:
    st.session_state['eda_report'] = None
    
def on_change_eda():
    st.session_state['eda_report'] = None

eda = st.selectbox('EDA', ['Pandas Profiling', 'Sweetviz'], on_change=on_change_eda())

# EDA Process-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Pandas Profiling============================================================================================================================================
    
if eda == 'Pandas Profiling':
    if st.session_state['eda_report'] is None:
        EDA_report= pp.ProfileReport(data, title="Pandas Profiling Report", explorative=True, dark_mode=True)
        st.session_state['eda_report'] = EDA_report
    
    st_profile_report(st.session_state['eda_report'])

# Pandas Profiling End========================================================================================================================================



# Sweetviz====================================================================================================================================================
def st_display_sweetviz(report_html):
    report_file = codecs.open(report_html, 'r')
    page = report_file.read()
    return page
    
    
if eda == 'Sweetviz':
    if st.session_state['eda_report'] == None:
        report = sv.analyze(data)
        report.show_html('report.html', open_browser=False)
        page = st_display_sweetviz('report.html')
        st.session_state['eda_report']==page
    
    
    components.html(st.session_state['eda_report'], width=800, height=800, scrolling=True)

# Sweetviz End================================================================================================================================================




# Train & Test Data Comparision===============================================================================================================================

# Label Encoder
label = LabelEncoder()

# Encoded DataFrame
final_data = data_label.apply(label.fit_transform)


# Spliting into X, y
X_label = final_data.iloc[:,1:]
y_label = final_data.iloc[:,0]

from sklearn.model_selection import train_test_split
# split X and y into training and testing sets
Xl_train, Xl_test, yl_train, yl_test = train_test_split(X_label, y_label, 
                                                        test_size = 0.2, random_state = 0)

compare_report = sv.compare([Xl_train, 'Train'], [Xl_test, 'Test'])
compare_report.show_html('compare.html', open_browser=False)

if st.button('Generate Comparison Report b/w Train & Test'):
    page_compare = st_display_sweetviz('compare.html')
    
    components.html(page_compare, width=800, height=800, scrolling=True)


# Train & Test Data Comparision End===========================================================================================================================














