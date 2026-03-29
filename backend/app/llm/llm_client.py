import os
from langchain_google_genai import GoogleGenerativeAI 
from dotenv import load_dotenv
load_dotenv()
class LLMClient:
    def __init__(self):
        self.llm = GoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
    
    def generate_response(self, prompt):
        return self.llm.invoke(prompt)


