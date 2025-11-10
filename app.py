from crew import crew_exocortex
def run (query):    
    results = crew_exocortex.kickoff(inputs = {"topic": query})
    return results
if __name__ == "__main__":
    run("I have a dataset at /Users/gunik/scientific_exocortex/Datasets/Iris.csv target variable is species, I want to train a decision tree model with hyperparameter tuning for max_depth and min_samples_split. Can you help me with that?")
