from crewai import Task
from Agents.for_agent import for_agent

support_task = Task(
    description="""
    Only proceed if the routing decision is 'route: supporting' or 'route: both'. If not, respond with 'skipping supporting task'. else, follow the steps below:
    You support the hypothesis in the user's question.
    
    Step 1: Rewrite {topic} into a search query that finds **supportive** literature published before the given date.
    Step 2: Use fetch_publications to find relevant papers.
    Step 3: Use the RagTool and Summarize the **evidence** supporting the claim. Cite papers directly. In your query always send similarity_threshold=0.8 and limit=5 to the RagTool.
    Step 4: Return a summary of the evidence supporting the claim, along with cited papers. and must include 'route: supporting' or 'route: both' at the top of the response(whichever applicable out of the two).
    """,
    agent=for_agent,
    expected_output="summarization of the evidence supporting the claim, cited papers",
    input_keys=["topic"],
    expected_output_key = "support_view",
    verbose=True
)