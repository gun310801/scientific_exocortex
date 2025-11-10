from crewai import Task
from Agents.analyzer_agent import analyzer_agent

analyze_task = Task(
    name="Task Analysis",
     description="You're tasked with routing the user's request {topic} based on its content.\n\n"
        "There are 3 possible categories, and you must identify which one applies:\n\n"

        "🔹 **1. route: supporting**\n"
        "- No dataset is mentioned (no file paths, data references).\n"
        "- The query only seeks conceptual explanation or theoretical help.\n"
        "- Flags: conceptual_flag = 1, dataset_flag = 0\n\n"

        "🔹 **2. route: exploration**\n"
        "- A dataset is clearly mentioned (e.g. `.csv`, `.json`, `.xlsx`, 'dataset', etc.).\n"
        "- The user is only asking for data analysis, not conceptual help.\n"
        "- Flags: conceptual_flag = 0, dataset_flag = 1\n\n"

        "🔹 **3. route: both**\n"
       " - A dataset is present **AND** the user is asking for conceptual insight or synthesis based on the data.\n"
       " - Flags: conceptual_flag = 1, dataset_flag = 1\n\n"

        "----------------------------------------\n"
        "🔎 Detection Tips:\n"
        "- Dataset indicators: `.csv`, `.json`, `.xlsx`, 'data', 'dataset', 'table', etc.\n"
        "- Conceptual indicators: 'explain', 'understand', 'insight', 'hypothesis', 'what does this mean', etc.\n\n"

        "📤 **Output Format:**\n"
        "Start your response with your routing decision on the first line:\n"
        "`route: supporting`, `route: exploration`, or `route: both`\n"
        "Then output this diagnostic block:\n"
        "```\n"
        "conceptual_flag: 0 or 1\n"
        "dataset_flag: 0 or 1\n"
        "```\n"
        "Then explain your reasoning briefly.\n\n"

        "⛔ Do NOT assume `route: supporting` by default. You must explicitly justify your decision based on both flags.\n\n"

        "✅ Example:\n"
        "`route: exploration`\n"
        "```\n"
        "conceptual_flag: 0\n"
        "dataset_flag: 1\n"
        "```\n"
        "The user asked for statistical summaries and provided a .csv file, but did not request conceptual understanding.",
    
    # description ="Analyze the user's input: to determine whether they require conceptual synthesis, data exploration, or both." \
    #  "There are three possible types of needs:\n"
    #     "1. Conceptual Synthesis (route: supporting) — when the user is asking for help understanding or generating conceptual insights. (dataset_flag: False)\n"
    #     "2. Data Exploration (route: exploration) — when the user provides a dataset or asks to analyze data (dataset_flag: True)\n"
    #     "3. Both (route: both) — when the task requires conceptual support and working with data. query demands both (dataset_flag: True)\n\n" \
    # "Your response should include the keyword ('route: supporting' or 'route: exploration' or 'route: both') clearly at the top. then explain your reasoning",
    expected_output="A clear routing decision ('route: supporting' or 'route: exploration' or 'route: both') and a short justification",
    agent=analyzer_agent,
    input_keys=["topic"],
    expected_output_key="route_decision",
    verbose=True)