import os
import PIL.Image
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

img = PIL.Image.open('image.JPG')
# print(img.show())

load_dotenv()

API_KEY = os.getenv("API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-pro')


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",api_key=API_KEY)

message = HumanMessage(
    content = [
        {
            "type":"text",
            "text":"write a short description about the product shown in image"
        },
        {
            "type": "image_url", 
            "image_url": "D:/Development/HP_Development/AI/RAG_ReadPDF/swetty.JPG"
        },
    ]
)

print(llm.invoke([message]).content)