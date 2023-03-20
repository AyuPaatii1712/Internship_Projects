# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 11:52:57 2022

@author: Nikhil
"""

# from tkinter import Button
import streamlit as st
import pickle
import numpy as np

def recommend_book(book):
    index = np.where(pt.columns==book)[0][0]
    similar_books = sorted(list(enumerate(similarity_score[index])),key=lambda x:x[1],reverse=True)[1:6]
    
    data = []
    for i in similar_books:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        
        data.append(item)
    return data
st.image('images.jpg')
st.header("Book Recommender System")
popular = pickle.load(open('popular.pkl','rb'))
books = pickle.load(open('books.pkl','rb'))
pt = pickle.load(open('pt.pkl','rb'))
similarity_score = pickle.load(open('similarity_score.pkl','rb'))

user_ID = pt.columns.values
image_url = popular['Image-URL-M'].tolist()
book_title = popular['Book-Title'].tolist() 
book_author = popular['Book-Author'].tolist()
total_ratings = popular['total Ratings'].tolist()
avg_ratings = popular['Total Average Ratings'].tolist()

st.sidebar.title("Top 50 Books")
if st.sidebar.button("SHOW"):
    col1,col2,col3,col4,col5=st.columns(5)
    with col1:
        st.image(image_url[0])
        st.text(book_author[0])
        st.text("Ratings:" + str(total_ratings[0]))
        st.text("Avg.Rating:" + str(round(avg_ratings[0],2)))
    with col2:
        st.image(image_url[1])
        st.text(book_author[1])
        st.text("Ratings:" + str(total_ratings[1]))
        st.text("Avg.Rating:" + str(round(avg_ratings[1],2)))
    with col3:
        st.image(image_url[2])
        st.text(book_author[2])
        st.text("Ratings:" + str(total_ratings[2]))
        st.text("Avg.Rating:" + str(round(avg_ratings[2],2)))
    with col4:
        st.image(image_url[3])
        st.text(book_author[3])
        st.text("Ratings:" + str(total_ratings[3]))
        st.text("Avg.Rating:" + str(round(avg_ratings[3],2)))
    with col5:
        st.image(image_url[4])
        st.text(book_author[4])
        st.text("Ratings:" + str(total_ratings[4]))
        st.text("Avg.Rating:" + str(round(avg_ratings[4],2)))
        
    col1,col2,col3,col4,col5=st.columns(5)
    with col1:
        st.image(image_url[5])
        st.text(book_author[5])
        st.text("Ratings:" + str(total_ratings[5]))
        st.text("Avg.Rating:" + str(round(avg_ratings[5],2)))
    with col2:
        st.image(image_url[6])
        st.text(book_author[6])
        st.text("Ratings:" + str(total_ratings[6]))
        st.text("Avg.Rating:" + str(round(avg_ratings[6],2)))
    with col3:
        st.image(image_url[7])
        st.text(book_author[7])
        st.text("Ratings:" + str(total_ratings[7]))
        st.text("Avg.Rating:" + str(round(avg_ratings[7],2)))
    with col4:
        st.image(image_url[8])
        st.text(book_author[8])
        st.text("Ratings:" + str(total_ratings[8]))
        st.text("Avg.Rating:" + str(round(avg_ratings[8],2)))
    with col5:
        st.image(image_url[9])
        st.text(book_author[9])
        st.text("Ratings:" + str(total_ratings[9]))
        st.text("Avg.Rating:" + str(round(avg_ratings[9],2)))

    col1,col2,col3,col4,col5=st.columns(5)
    with col1:
        st.image(image_url[10])
        st.text(book_author[10])
        st.text("Ratings:" + str(total_ratings[10]))
        st.text("Avg.Rating:" + str(round(avg_ratings[10],2)))
    with col2:
        st.image(image_url[11])
        st.text(book_author[11])
        st.text("Ratings:" + str(total_ratings[11]))
        st.text("Avg.Rating:" + str(round(avg_ratings[11],2)))
    with col3:
        st.image(image_url[12])
        st.text(book_author[12])
        st.text("Ratings:" + str(total_ratings[12]))
        st.text("Avg.Rating:" + str(round(avg_ratings[12],2)))
    with col4:
        st.image(image_url[13])
        st.text(book_author[13])
        st.text("Ratings:" + str(total_ratings[13]))
        st.text("Avg.Rating:" + str(round(avg_ratings[13],2)))
    with col5:
        st.image(image_url[14])
        st.text(book_author[14])
        st.text("Ratings:" + str(total_ratings[14]))
        st.text("Avg.Rating:" + str(round(avg_ratings[14],2)))  
    
    col1,col2,col3,col4,col5=st.columns(5)
    with col1:
        st.image(image_url[15])
        st.text(book_author[15])
        st.text("Ratings:" + str(total_ratings[15]))
        st.text("Avg.Rating:" + str(round(avg_ratings[15],2)))
    with col2:
        st.image(image_url[16])
        st.text(book_author[16])
        st.text("Ratings:" + str(total_ratings[16]))
        st.text("Avg.Rating:" + str(round(avg_ratings[16],2)))
    with col3:
        st.image(image_url[17])
        st.text(book_author[17])
        st.text("Ratings:" + str(total_ratings[17]))
        st.text("Avg.Rating:" + str(round(avg_ratings[17],2)))
    with col4:
        st.image(image_url[18])
        st.text(book_author[18])
        st.text("Ratings:" + str(total_ratings[18]))
        st.text("Avg.Rating:" + str(round(avg_ratings[18],2)))
    with col5:
        st.image(image_url[19])
        st.text(book_author[19])
        st.text("Ratings:" + str(total_ratings[19]))
        st.text("Avg.Rating:" + str(round(avg_ratings[19],2))) 
        
    col1,col2,col3,col4,col5=st.columns(5)
    with col1:
        st.image(image_url[20])
        st.text(book_author[20])
        st.text("Ratings:" + str(total_ratings[20]))
        st.text("Avg.Rating:" + str(round(avg_ratings[20],2)))
    with col2:
        st.image(image_url[21])
        st.text(book_author[21])
        st.text("Ratings:" + str(total_ratings[21]))
        st.text("Avg.Rating:" + str(round(avg_ratings[21],2)))
    with col3:
        st.image(image_url[22])
        st.text(book_author[22])
        st.text("Ratings:" + str(total_ratings[22]))
        st.text("Avg.Rating:" + str(round(avg_ratings[22],2)))
    with col4:
        st.image(image_url[23])
        st.text(book_author[23])
        st.text("Ratings:" + str(total_ratings[23]))
        st.text("Avg.Rating:" + str(round(avg_ratings[23],2)))
    with col5:
        st.image(image_url[24])
        st.text(book_author[24])
        st.text("Ratings:" + str(total_ratings[24]))
        st.text("Avg.Rating:" + str(round(avg_ratings[24],2))) 
            
    col1,col2,col3,col4,col5=st.columns(5)
    with col1:
        st.image(image_url[25])
        st.text(book_author[25])
        st.text("Ratings:" + str(total_ratings[25]))
        st.text("Avg.Rating:" + str(round(avg_ratings[25],2)))
    with col2:
        st.image(image_url[26])
        st.text(book_author[26])
        st.text("Ratings:" + str(total_ratings[26]))
        st.text("Avg.Rating:" + str(round(avg_ratings[26],2)))
    with col3:
        st.image(image_url[27])
        st.text(book_author[27])
        st.text("Ratings:" + str(total_ratings[27]))
        st.text("Avg.Rating:" + str(round(avg_ratings[27],2)))
    with col4:
        st.image(image_url[28])
        st.text(book_author[28])
        st.text("Ratings:" + str(total_ratings[28]))
        st.text("Avg.Rating:" + str(round(avg_ratings[28],2)))
    with col5:
        st.image(image_url[29])
        st.text(book_author[29])
        st.text("Ratings:" + str(total_ratings[29]))
        st.text("Avg.Rating:" + str(round(avg_ratings[29],2))) 
    col1,col2,col3,col4,col5=st.columns(5)
    with col1:
        st.image(image_url[30])
        st.text(book_author[30])
        st.text("Ratings:" + str(total_ratings[30]))
        st.text("Avg.Rating:" + str(round(avg_ratings[30],2)))
        with col2:
            st.image(image_url[31])
            st.text(book_author[31])
            st.text("Ratings:" + str(total_ratings[31]))
            st.text("Avg.Rating:" + str(round(avg_ratings[31],2)))
        with col3:
            st.image(image_url[32])
            st.text(book_author[32])
            st.text("Ratings:" + str(total_ratings[32]))
            st.text("Avg.Rating:" + str(round(avg_ratings[32],2)))
        with col4:
            st.image(image_url[33])
            st.text(book_author[33])
            st.text("Ratings:" + str(total_ratings[33]))
            st.text("Avg.Rating:" + str(round(avg_ratings[33],2)))
        with col5:
            st.image(image_url[34])
            st.text(book_author[34])
            st.text("Ratings:" + str(total_ratings[34]))
            st.text("Avg.Rating:" + str(round(avg_ratings[34],2)))
            
    col1,col2,col3,col4,col5=st.columns(5)
    with col1:
            st.image(image_url[35])
            st.text(book_author[35])
            st.text("Ratings:" + str(total_ratings[35]))
            st.text("Avg.Rating:" + str(round(avg_ratings[35],2)))
    with col2:
            st.image(image_url[36])
            st.text(book_author[36])
            st.text("Ratings:" + str(total_ratings[36]))
            st.text("Avg.Rating:" + str(round(avg_ratings[36],2)))
    with col3:
              st.image(image_url[37])
              st.text(book_author[37])
              st.text("Ratings:" + str(total_ratings[37]))
              st.text("Avg.Rating:" + str(round(avg_ratings[37],2)))
    with col4:
              st.image(image_url[38])
              st.text(book_author[38])
              st.text("Ratings:" + str(total_ratings[38]))
              st.text("Avg.Rating:" + str(round(avg_ratings[38],2)))
    with col5:
              st.image(image_url[39])
              st.text(book_author[39])
              st.text("Ratings:" + str(total_ratings[39]))
              st.text("Avg.Rating:" + str(round(avg_ratings[39],2)))    
    col1,col2,col3,col4,col5=st.columns(5)
    with col1:
            st.image(image_url[40])
            st.text(book_author[40])
            st.text("Ratings:" + str(total_ratings[40]))
            st.text("Avg.Rating:" + str(round(avg_ratings[40],2)))
    with col2:
            st.image(image_url[41])
            st.text(book_author[41])
            st.text("Ratings:" + str(total_ratings[41]))
            st.text("Avg.Rating:" + str(round(avg_ratings[41],2)))
    with col3:
            st.image(image_url[42])
            st.text(book_author[42])
            st.text("Ratings:" + str(total_ratings[42]))
            st.text("Avg.Rating:" + str(round(avg_ratings[42],2)))
    with col4:
            st.image(image_url[43])
            st.text(book_author[43])
            st.text("Ratings:" + str(total_ratings[43]))
            st.text("Avg.Rating:" + str(round(avg_ratings[43],2)))
    with col5:
            st.image(image_url[44])
            st.text(book_author[44])
            st.text("Ratings:" + str(total_ratings[44]))
            st.text("Avg.Rating:" + str(round(avg_ratings[44],2)))   
            
    col1,col2,col3,col4,col5=st.columns(5)
    with col1:
            st.image(image_url[45])
            st.text(book_author[45])
            st.text("Ratings:" + str(total_ratings[45]))
            st.text("Avg.Rating:" + str(round(avg_ratings[45],2)))
    with col2:
            st.image(image_url[46])
            st.text(book_author[46])
            st.text("Ratings:" + str(total_ratings[46]))
            st.text("Avg.Rating:" + str(round(avg_ratings[46],2)))
    with col3:
            st.image(image_url[47])
            st.text(book_author[47])
            st.text("Ratings:" + str(total_ratings[47]))
            st.text("Avg.Rating:" + str(round(avg_ratings[47],2)))
    with col4:
            st.image(image_url[48])
            st.text(book_author[48])
            st.text("Ratings:" + str(total_ratings[48]))
            st.text("Avg.Rating:" + str(round(avg_ratings[48],2)))
    with col5:
            st.image(image_url[49])
            st.text(book_author[49])
            st.text("Ratings:" + str(total_ratings[49]))
            st.text("Avg.Rating:" + str(round(avg_ratings[49],2)))     
            
            
st.sidebar.title("Recommend Books")
selected_book = st.sidebar.selectbox("Type or select a book from the dropdown",user_ID)
if st.sidebar.button("Recommend Me"):
    moviee = recommend_book(selected_book)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(moviee[0][2])
        st.text(moviee[0][0])
        st.text(moviee[0][1])  
    with col2:
        st.image(moviee[1][2])
        st.text(moviee[1][0])
        st.text(moviee[1][1])
    with col3:
        st.image(moviee[2][2])
        st.text(moviee[2][0])
        st.text(moviee[2][1])
    with col4:
        st.image(moviee[3][2])
        st.text(moviee[3][0])
        st.text(moviee[3][1])
    with col5:
        st.image(moviee[4][2])
        st.text(moviee[4][0])
        st.text(moviee[4][1])
        
        
        
 

 