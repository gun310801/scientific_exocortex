from crew import crew_exocortex
def run (query):    
    results = crew_exocortex.kickoff(inputs = {"topic": query})
    return results
if __name__ == "__main__":
    run("which LLM makes the best scientific agents")
#optional question to ask:
#Analyze the Iris dataset with a decision tree, logistic regression, and SVM. Compare the results and suggest which model performs best for this dataset
#Application of Machine Learning in scientific research: Analyze the Iris dataset .
#You can upload other datasets to the Datasets/ folder to run different experiments.