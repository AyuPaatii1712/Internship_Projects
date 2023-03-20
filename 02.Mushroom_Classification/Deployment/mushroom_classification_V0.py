# Import the Libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
#import numpy as np


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
     

# Data Preprocessing i.e. Label Encoding--------------------------------------------------------------------------------------------------------------------------------------


#from sklearn.preprocessing import LabelEncoder

# Label Encoder
label = LabelEncoder()

# Encoded DataFrame
final_data = data_label.apply(label.fit_transform)
st.subheader('Encoded DataSet use for Train and Test')
st.dataframe(final_data)

# Data Encoding End-----------------------------------------------------------------------------------------------------------------------------------------------------------

# Model Building--------------------------------------------------------------------------------------------------------------------------------------------------------------

# Import necessary Libraries
from sklearn.tree import  DecisionTreeClassifier
from sklearn.model_selection import train_test_split #, GridSearchCV
from sklearn.metrics import classification_report , accuracy_score , f1_score, confusion_matrix
        
        
# Spliting into X, y
X_label = final_data.iloc[:,1:]
y_label = final_data.iloc[:,0]

# split X and y into training and testing sets
Xl_train, Xl_test, yl_train, yl_test = train_test_split(X_label, y_label, 
                                                        test_size = 0.2, random_state = 0)

# Model Training
tre = DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_split=2, random_state=0)
tre.fit(Xl_train, yl_train)

# Model Building End----------------------------------------------------------------------------------------------------------------------------------------------------------

# Model Validation------------------------------------------------------------------------------------------------------------------------------------------------------------

# Training Validation_______________________________________________________________________________________

#Predict for X dataset
y_train_predict_tree = tre.predict(Xl_train)

# Training Accuracy and F1-score
train_acc_score = accuracy_score(yl_train, y_train_predict_tree)
train_f1_score = f1_score(yl_train, y_train_predict_tree)

st.subheader('Training Accuracy')
st.write('Train Accuracy Score : ' , train_acc_score)
st.write('Train F1 Score : ' , train_f1_score)

# print classification report
st.text('Model Report on Training DataSet:\n ' 
        +classification_report(yl_train, y_train_predict_tree, digits=4))


# Confusion Matrix for Train Data
cm = pd.DataFrame(confusion_matrix(yl_train,y_train_predict_tree), 
                                   columns=['Edible', 'Poisonous'], index=['Edible', 'Poisonous'])

sns.set_theme(style='dark')
sns.set(rc={'axes.facecolor':'#282828', 'figure.facecolor':'#282828'})

fig, ax = plt.subplots()
sns.heatmap(cm,annot=True,fmt='.0f', ax=ax)
#ax.tick_params(grid_color='r', labelcolor='r', color='r')
    
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.title.set_color('white')
ax.tick_params(colors='white')
ax.figure.axes[-1].tick_params(colors='white')
plt.show()
st.write(fig)

st.text("")
st.text("")
st.write('#')

# Training Validation End___________________________________________________________________________________

        
# Testing Validation________________________________________________________________________________________

#Predict for X dataset       
y_test_predict_tree = tre.predict(Xl_test)

# Testing Accuracy and F1-Score
test_acc_score = accuracy_score(yl_test, y_test_predict_tree)
test_f1_score = f1_score(yl_test, y_test_predict_tree)
st.subheader('Testing Accuracy')
st.write('Test Accuracy Score : ' , test_acc_score)
st.write('Test F1 Score : ' , test_f1_score)

# print classification report
st.text('Model Report on Testing DataSet:\n ' 
        +classification_report(yl_test, y_test_predict_tree, digits=4))       
        

# Confusion Matrix for Test Data      
cm = pd.DataFrame(confusion_matrix(yl_test,y_test_predict_tree), 
                                   columns=['Edible', 'Poisonous'], index=['Edible', 'Poisonous'])

sns.set_theme(style='dark')
sns.set(rc={'axes.facecolor':'#282828', 'figure.facecolor':'#282828'})

fig, ax = plt.subplots()
sns.heatmap(cm,annot=True,fmt='.0f', ax=ax)
#ax.tick_params(grid_color='r', labelcolor='r', color='r')
    
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.title.set_color('white')
ax.tick_params(colors='white')
ax.figure.axes[-1].tick_params(colors='white')
plt.show()
st.write(fig)        

# Test Validation End_______________________________________________________________________________________
        
# Model Validation End-------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        
        
        
        
        
        
        
        