from crewai import Agent, LLM
import os
from dotenv import load_dotenv
load_dotenv()

llm = LLM(
    model="gemini/gemini-2.5-flash",
    temperature=0,
    api_key=os.getenv("GEMINI_API_KEY"),
)

final_agent = Agent(
    llm=llm,
    name = "Final Synthesizer Agent",
    role = "Research Synthesizer and Final Report Writer on the basis of previous debate and exploration",
     goal="Compare  contradictions, gaps, or unexplored dimensions with the explored data and write a final report that synthesizes the insights from the debate and data exploration. The report should be well-structured, clearly written, and include citations to the sources used.",
    backstory="You synthesize insights by reading through the debate and data exploration results, identifying where the truth breaks down or has yet to be found, and writing a final report that summarizes the findings in a clear and scientific manner.",
    tools=[],
    verbose=True,
    cache=False
)
