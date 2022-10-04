#import streamlit
import streamlit as sl

st.write(
    """
    #Generate new text from dead people
"""
)

st.write(
    """
    This is a simple AI that can write like your favorite Project Gutenburg authors. To use it, simply put in a link to an author who is on Project Gutenburg
"""
)

book_url = st.text_input(label = "Input a link to an author on Project Gutenburg")

book_id = int(book_url.split('/')[4])

