from crewai import Task
from Agents.final_synthesizer_agent import final_agent

final_synthesizer_task = Task( 
     description="""
    Only proceed if the routing decision is  'route: both'. If routing decision is 'route: supporting', respond with "synthesis_view" and if routing decisison is 'route: exploration' respond with "smartml_response".else, Analyze the outputs from synthesis_view and smartml_response.
    Based on the data and insights from the debate and the smartml_response, write a final report that synthesizes the insights from both. The report should be well-structured, clearly written, and include citations to the sources used.  
    Highlight where the evidence **conflicts**, where it **aligns**, and what remains **uncertain**.
    Conclude with a suggestion of next experiment.
    """,
    agent=final_agent,
    expected_output = "summarization of output from both agents, highlighting conflicts and gap for research include **alignment**, **conflicts**, **uncertainties**, **assumptions**, **conflict insights**, **next experiment**",
    input_keys=["synthesis_view","smartml_response"]
)