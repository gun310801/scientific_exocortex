from crewai import Task
from Agents.against_agent import against_agent

against_task = Task(
    description="""
    You oppose the hypothesis in the user's question.
    Only proceed if the routing decision is 'route: supporting' or 'route: both'. If not, respond with 'skipping supporting task'. else, follow the steps below:
    Step 1: Rewrite {topic} into a search query that finds **opposing** literature published .
    Step 2: Use fetch_publications_against tool to find relevant papers.
    Step 3: Use the RagTool and Summarize the **evidence** opposing the claim. Cite papers directly. In your query always send similarity_threshold=0.8 and limit=5 to the RagTool.
    Step 4: Return a summary of the evidence opposing the claim, along with cited papers. and must include 'route: supporting' or 'route: both' at the top of the response(whichever applicable out of the two).
    """,
    agent=against_agent,
    expected_output="summarization of the evidence opposing the claim, cited papers",
    input_keys=["topic"],
    expected_output_key = "against_view",
    verbose=True
)