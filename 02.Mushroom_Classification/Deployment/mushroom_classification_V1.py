# Import the Libraries
import streamlit as st
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.preprocessing import LabelEncoder
#import numpy as np

st.set_page_config(page_title='Mushroom Classification', page_icon='mushroom1.jpg')
st.image('mushroom31.jpg')
st.header('Mushroom Classification')
st.write('---')

radio = st.radio(label = "What you want to do?", options = ['Train & Test the Model','Classify using Model'])
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)


# Input DataSet Used to Train the Model-----------------------------------------------------------------------------------------------------------------------------------------
st.sidebar.header('User Input Parameters')
st.sidebar.write('---')
st.sidebar.write('**Classify based on which DataSet**')
data_file = st.sidebar.selectbox(
    label ="Classify based on ",
    options=['Default Data','Upload the Data'])

# To use default file for training the model
if data_file == 'Default Data':
    st.subheader('Input DataFrame')
    data = pd.read_csv('mushrooms.csv')
    st.dataframe(data)

# Upload another file
if data_file == 'Upload the Data':
    file = st.sidebar.file_uploader('Upload DataSet In "csv" formate', type = 'csv')
    
    # If File is not Uploaded
    if file == None:
        st.error('Please Upload the file')
        st.stop()
        
    else:
        data = pd.read_csv(file)
        data1 = pd.read_csv('mushrooms.csv')
        st.subheader('Input DataSet')
        st.dataframe(data)
        
        # Columns use to Train the Model
        use_cols = ['class','spore-print-color','gill-color','gill-size','stalk-root',
                    'habitat','stalk-shape','odor','population']
        
        # What to do if Columns is Present OR Not Present
        try:
            data_label = data[use_cols]

        except:
            st.error('Please Upload the correct file, your file contain below columns and in the same order')
            st.write(pd.DataFrame(use_cols, columns=['columns']))
            st.stop()

# Input DataSet is Taken--------------------------------------------------------------------------------------------------------------------------------------------------------

# File to be Classify-----------------------------------------------------------------------------------------------------------------------------------------------------------

# Upload that file you want to be Classify
st.sidebar.write('**Upload the Data you want to Classify**')
test_file = st.sidebar.file_uploader('Upload DataSet In "csv" formate', type = 'csv')

# If File is not Uploaded
if test_file == None:
    st.error('Please Upload the file')
    st.stop()
    
# If File is Uploaded
else:
    test_data = pd.read_csv(test_file)
    st.subheader('DataSet to be Classify')
    st.dataframe(test_data)

# Column that are used for Test DataSet
use_test_cols = ['spore-print-color','gill-color','gill-size','stalk-root',
            'habitat','stalk-shape','odor','population']

# What to do if Columns is Present OR Not Present
try:
    test_data = test_data[use_test_cols]

except:
    st.error('File to be Classify is not correct, Please upload the correct file and your file contain below columns')
    st.write(pd.DataFrame(use_test_cols, columns=['columns']))
    st.stop()
    
# File to be Classify End-------------------------------------------------------------------------------------------------------------------------------------------------------

# Model Building and Training---------------------------------------------------------------------------------------------------------------------------------------------------

from sklearn.tree import  DecisionTreeClassifier

# Columns use to Train the Model
use_cols = ['class','spore-print-color','gill-color','gill-size','stalk-root',
            'habitat','stalk-shape','odor','population']

# Final Training DataFrame
data_label = data[use_cols]

# Label Encoding
label = LabelEncoder()
final_data = data_label.apply(label.fit_transform)

# Spliting into X, y
X_label = final_data.iloc[:,1:]
y_label = final_data.iloc[:,0]

# Model Training
tre = DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_split=2, random_state=0)
tre.fit(X_label, y_label)

# Model Building End------------------------------------------------------------------------------------------------------------------------------------------------------------



# Final Prediction--------------------------------------------------------------------------------------------------------------------------------------------------------------

# Dictionary use for Mapping the Test Data besed on Train Data
dt_class = {0:'Edible', 1:'Poisonous'}
dt_spore_print_color = {'k': 2, 'n': 3, 'u': 6, 'h': 1, 'w': 7, 'r': 5, 'o': 4, 'y': 8, 'b': 0}
dt_gill_color = {'k': 4, 'n': 5, 'g': 2, 'p': 7, 'w': 10, 'h': 3, 'u': 9, 'e': 1, 'b': 0, 'r': 8, 'y': 11, 'o': 6}
dt_gill_size = {'n': 1, 'b': 0}
dt_stalk_root = {'e': 3, 'c': 2, 'b': 1, 'r': 4, '?': 0}
dt_habitat = {'u': 5, 'g': 1, 'm': 3, 'd': 0, 'p': 4, 'w': 6, 'l': 2}
dt_stalk_shape = {'e': 0, 't': 1}
dt_odor = {'p': 6, 'a': 0, 'l': 3, 'n': 5, 'f': 2, 'c': 1, 'y': 8, 's': 7, 'm': 4}
dt_population = {'s': 3, 'n': 2, 'a': 0, 'v': 4, 'y': 5, 'c': 1}


# Single Dictionary with Key name as Column name
map_label = {'spore-print-color':dt_spore_print_color, 'gill-color':dt_gill_color, 'gill-size':dt_gill_size, 
             'stalk-root':dt_stalk_root, 'habitat':dt_habitat, 'stalk-shape':dt_stalk_shape, 'odor':dt_odor, 
             'population':dt_population}


# Label Encoding done with "map" command and map_label dictionary
test_label = pd.DataFrame()
for col in use_test_cols:
    test_label[col] = test_data[col].map(map_label[col])

#st.dataframe(test_label)

# Test Prediction
y_pred_tree = tre.predict(test_label)

# Prediction DataFrame with Test Input
pred_data = test_data.copy()
pred_data['class'] = y_pred_tree
pred_data['class'] = pred_data['class'].map({0:'Edible', 1:'Poisonous'})

# Fuction define to color the dataframe
def color_df(clas):
    if clas == 'Poisonous':
        color = 'tomato'
    elif clas == 'Edible':
        color = 'green'
    else:
        color = 'dimgrey'
        
    return f'background-color: {color}'


# Final DataFrame
st.subheader('Classified Data or Output')
st.dataframe(pred_data.style.applymap(color_df, subset=['class']))



# Value Counts of Final Dataframe
dt = {'Mushroom_Classification':pred_data['class'].value_counts().index.tolist(), 
      'Counts':pred_data['class'].value_counts().values.tolist()}
value_counts = pd.DataFrame(dt)
st.subheader('Value Counts of Classified Data')
st.dataframe(value_counts.style.applymap(color_df, subset=['Mushroom_Classification']))

# Final Pridiction Over---------------------------------------------------------------------------------------------------------------------------------------------------------







