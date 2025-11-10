from crewai import Agent, LLM
import os
from dotenv import load_dotenv
load_dotenv()

llm = LLM(
    model="gemini/gemini-2.5-flash",
    temperature=0,
    api_key=os.getenv("GEMINI_API_KEY"),
)

debate_agent = Agent(
    llm=llm,
    name = "Debate Agent",
    role = "Research Synthesizer",
     goal="Compare opposing viewpoints and surface contradictions, gaps, or unexplored dimensions.",
    backstory="You synthesize insights by reading through competing claims and identifying where the truth breaks down or has yet to be found.",
    tools=[],
    verbose=True,
    cache=False
)
