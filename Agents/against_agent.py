from crewai import Agent, LLM

from Tools.find_publication_against import fetch_publications_against
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

llm = LLM(
    model="gemini/gemini-2.5-flash",
    temperature=0,
    api_key=os.getenv("GEMINI_API_KEY"),
)

against_agent = Agent(
    llm=llm,
    tools=[fetch_publications_against],
    name = "Against Agent",
    role = "opposing Researcher",
    goal = "Construct a persuasive, evidence-based argument opposing the given hypothesis using credible, peer-reviewed research literature.",
    backstory ="You are convinced the hypothesis is not valid. Your role is to gather, analyze, and synthesize high-quality scholarly sources to build a strong, well-cited case demonstrating its credibility and theoretical soundness.",
    verbose=True,
    cache=False
)