import streamlit as st

from llm import Extractor

from copy import deepcopy
import pandas as pd


st.title("Echo Bot")

# Initialize chat history
if "messages" not in st.session_state:
    
    st.session_state.messages = []
    st.session_state.is_first = True
    st.session_state.next_missing_data = None
    st.session_state.data = {}

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# React to user input
if prompt := st.chat_input("What is up?"):

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    response = Extractor.extract(prompt, st.session_state)
    #response = {"Tacos" : "True", "Burriots" : None}
    st.session_state.data = response
    
    #st.session_state.data = response
    
    
    st.session_state.is_first = False

    # Display assistant response in chat message container
    # with st.chat_message("assistant"):
    #     st.markdown(response)
    # Add assistant response to chat history
    #st.session_state.messages.append({"role": "assistant", "content": response})
    
        

if st.session_state.is_first == False:
    
    next_missing_data = Extractor.find_next_missing_data(st.session_state.data)
    
    if next_missing_data == None:
        st.markdown("Building has all required information.")
        st.write(pd.DataFrame({'Fields' : st.session_state.data.keys(), "Values" : st.session_state.data.values()}).set_index("Fields"))
    else:
        with st.chat_message("assistant"):
            st.markdown( "Please specify the value for: " + next_missing_data)
            st.session_state.next_missing_data = next_missing_data
        st.session_state.messages.append({"role" : "assistant", "content" : "Please specify the value for: " + next_missing_data})



