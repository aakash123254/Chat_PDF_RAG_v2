import streamlit as st
import google.generativeai as genai

def test_gemini_api():
    """Test the Gemini Flash API with a sample query."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    query = "Where is Finland?"
    response = model.generate_content(query)
    return response.text if response else "No response from Gemini."

# Debugging: Test Gemini API
st.write("Testing Gemini API...")
st.write(test_gemini_api())