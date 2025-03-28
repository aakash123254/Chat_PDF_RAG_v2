import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Function to list all available models and their supported methods
def list_available_models():
    try:
        print("Listing available models:")
        for model in genai.list_models():
            print(f"Model Name: {model.name}")
            
            # Check supported generation methods
            if hasattr(model, "supported_generation_methods"):
                print(f"Supported Generation Methods: {model.supported_generation_methods}")
            else:
                print("No supported generation methods listed.")
            
            print("-" * 50)
    except Exception as e:
        print("Error:", str(e))

if __name__ == "__main__":
    list_available_models()