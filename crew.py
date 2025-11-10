from crewai import Crew
from Tasks.analyzing_task import analyze_task
from Tasks.support_task import support_task
from Tasks.against_task import against_task
from Tasks.synthesizer_task import synthesizer_task
from Tasks.ML_task import smartml_task
from Tasks.final_synthesizer_task import final_synthesizer_task
from Agents.debate_agent import debate_agent
from Agents.smartml_agent import smartml_agent
from Agents.analyzer_agent import analyzer_agent
from Agents.for_agent import for_agent  
from Agents.against_agent import against_agent
from Agents.final_synthesizer_agent import final_agent

import os
from dotenv import load_dotenv
load_dotenv()

crew_exocortex = Crew(
    name="Scientific Exocortex",
    agents=[analyzer_agent, for_agent, against_agent, debate_agent,smartml_agent,final_agent],
    tasks=[analyze_task, support_task, against_task, synthesizer_task, smartml_task, final_synthesizer_task],
    verbose=True)