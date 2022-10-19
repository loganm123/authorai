#import streamlit
from functions import create_training_set
import streamlit as st
from functions import *
from knockknock import discord_sender
from webhook_url import webhook_url
import gutenbergpy.textget

st.write(
    """
    # Generate new text from dead people
"""
)

st.write(
    """
    This is a simple AI that can write like your favorite Project Gutenburg authors. To use it, simply put in a link to an author who is on Project Gutenburg
"""
)

book_url = st.text_input(label = "Input a link to an author on Project Gutenburg",value = "https://www.gutenberg.org/ebooks/69087")


#book_id = int(book_url.split('/')[4])

book_id = st.text_input(label="enter the book id here")

#book_id = 1000

st.write(book_id)

st.write("""
    Enter in a word to start the prompt
""")

texts = st.text_input(label = "Input in a word", value = "p")

encoded = get_clean_tokenize_encode(book_id)

dataset, max_id = create_training_set(encoded)

model = create_model(max_id)

history = train_model(model, dataset, 10)

new_text = complete_text(texts,temperature = 0.3)

