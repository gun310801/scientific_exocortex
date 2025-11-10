from crewai import Agent, LLM
import os
from dotenv import load_dotenv
load_dotenv()

llm = LLM(
    model="gemini/gemini-2.5-flash",
    temperature=0,
    api_key=os.getenv("GEMINI_API_KEY"),
)

analyzer_agent = Agent(
    llm=llm,
    name = "Initial Analyzing Agent",
    role = "Task Analyzer",
    goal = "Decide wether the user wants conceptual synthesis OR data exploration OR both",
    backstory ="You are an expert in analyzing user tasks. Your role is to determine whether the user requires conceptual synthesis, data exploration, or both.",
    verbose=True,
    cache=False
)