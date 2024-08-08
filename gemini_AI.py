import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI



load_dotenv()

API_KEY = os.getenv("API_KEY")

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel('gemini-pro')


# ask_question = input("How Can i help you !: ")
# response = model.generate_content(ask_question)
# print(response.text)

llm = ChatGoogleGenerativeAI(model="gemini-pro",api_key=API_KEY)
result = llm.invoke("i want to open a resturant for italian food. suggest a fancy name for this")
print(result.content)
