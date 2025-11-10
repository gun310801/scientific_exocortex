from crewai import Agent, LLM
from crewai_tools import RagTool
import os
from Tools.runner import train_ml_model
from dotenv import load_dotenv
load_dotenv()

rag = RagTool()
rag.add( data_type="web_page", url="https://scikit-learn.org/stable/modules/linear_model.html")
rag.add(data_type="web_page", url="https://scikit-learn.org/stable/modules/svm.html")
rag.add(data_type="web_page", url = "https://scikit-learn.org/stable/modules/tree.html")

llm = LLM(
    model="gemini/gemini-2.5-flash",
    temperature=0,
    api_key=os.getenv("GEMINI_API_KEY"),
)


smartml_agent = Agent(
    llm=llm,
    name="SmartML Agent",
    role="Machine Learning Expert",
    goal="Assist users in understanding and applying machine learning concepts, algorithms, and best practices.",
    backstory="You are a machine learning expert with extensive knowledge of algorithms and explaining concepts and applying models on datasets.",
    tools=[rag,train_ml_model],
    verbose=True,
    cache=False
    
)
