from crewai import Agent, LLM

from Tools.find_publication_support import fetch_publications
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

llm = LLM(
    model="gemini/gemini-2.5-flash",
    temperature=0,
    api_key=os.getenv("GEMINI_API_KEY"),
)

# rag_tool = RagTool(similarity_threshold=0.7,  # Adjust this value (0.0 to 1.0)
#     limit=5 )
# download_dir = Path(__file__).resolve().parent.parent / "Tools" / "downloaded_papers"

# if download_dir.exists():
#     for pdf_path in download_dir.glob("*.pdf"):
#         rag_tool.add(data_type="file", path=str(pdf_path))
#         print(f"Added {pdf_path} to RagTool.")

for_agent = Agent(
    llm=llm,
    tools=[fetch_publications],
    name = "For Agent",
    role = "Supporting Researcher",
    goal = "Construct a persuasive, evidence-based argument supporting the given hypothesis using credible, peer-reviewed research literature.",
    backstory ="You are convinced the hypothesis is valid. Your role is to gather, analyze, and synthesize high-quality scholarly sources to build a strong, well-cited case demonstrating its credibility and theoretical soundness.",
    verbose=True,
    cache=False
)
