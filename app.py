from crew import crew_exocortex
def run (query):    
    results = crew_exocortex.kickoff(inputs = {"topic": query})
    return results
if __name__ == "__main__":
    run("What are the latest advancements in machine learning for natural language processing? Apply a machine learning model to a dataset and explain the results.")
