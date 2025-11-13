from crewai import Task
from Agents.debate_agent import debate_agent

synthesizer_task = Task( 
     description="""
     Only proceed if the routing decision is 'route: supporting' or 'route: both'. If not, respond with 'skipping supporting task'. else, Analyze the outputs from support_view and oppose_view.
    Highlight where the evidence **conflicts**, where it **aligns**, and what remains **uncertain**.
    If both views rely on different datasets, benchmarks, or assumptions — surface that.
    Conclude with a **conflict insight** and suggest next experiment.
    """,
    agent=debate_agent,
    expected_output = "summarization of output from both agents, highlighting conflicts and gap for research include **alignment**, **conflicts**, **uncertainties**, **assumptions**, **conflict insights**, **next experiment**",
    input_keys=["support_view", "oppose_view"],
    expected_output_key = "synthesis_view",
    verbose=True
)