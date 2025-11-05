import streamlit as st
st.title("My Streamlit App")
st.header("header part")
st.text("this part contains basic text")

options = st.radio("Select an option:", ['M', 'E', 'N'])
st.write("you selected", options)

#dropdowm box
city = st.selectbox("select city", ['Delhi', 'Mumbai', 'Bangalore'])
st.write(f"you chose {city}")

name = st.text_input("Enter your name:")
st.write("Hello", name)
