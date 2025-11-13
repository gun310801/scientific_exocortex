# Scientific Exocortex

Scientific Exocortex is a CrewAI-powered research copilot that orchestrates multiple specialized AI agents to analyze a user question, gather supporting and opposing literature from arXiv, optionally run classical ML experiments on tabular data, debate the findings, and deliver a final synthesis. The project is designed as a template for multi-agent scientific workflows that combine retrieval, reasoning, and lightweight modeling.

## Features
- **Task routing:** An analyzer agent inspects each query and decides whether the request is conceptual (`route: supporting`), data-driven (`route: exploration`), or needs both.
- **Literature retrieval:** Dedicated "for" and "against" agents call arXiv via custom tools to download PDFs, index them with `RagTool`, and cite the evidence.
- **Debate & synthesis:** A debate agent compares the two viewpoints, and a final synthesizer reconciles debate + ML findings into a structured report with conflicts, alignments, uncertainties, and next experiments.
- **SmartML assistant:** A natural-language ML copilot that answers theory questions, emits JSON training specs, ML  models via `train_ml_model`, supports hyperparameter tuning, model comparison, and evaluation, and keeps a human-in-the-loop conversation so you can iteratively adjust parameters or clarify requests.

## Repository Structure
```
scientific_exocortex/
├─ app.py                     # Simple entrypoint that runs the crew
├─ crew.py                    # Crew definition: agents + tasks wiring
├─ Agents/                    # Individual CrewAI agent definitions
├─ Tasks/                     # CrewAI task prompts and routing guards
├─ Tools/
│   ├─ find_publication_support.py   # arXiv tool for supportive evidence
│   ├─ find_publication_against.py   # arXiv tool for opposing evidence
│   └─ runner.py                     # train_ml_model tool implementation
├─ Datasets/                  # Tabular datasets for SmartML (e.g., Iris.csv)
├─ JSONs/
│   ├─ model_parameters.json  # Default hyperparameters per model type
│   └─ sample.json            # Example SmartML specification payload
├─ requirements.txt           # Python dependencies
└─ venv/                      # (Optional) local virtual environment
```

## Prerequisites
- Python 3.11+
- An API key for **Gemini** (set as `GEMINI_API_KEY` in your env or `.env` file)
- (Recommended) Unix-like shell for the provided commands

## Setup
```bash
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Create .env with required secrets
cat <<'EOF' > .env
GEMINI_API_KEY=your_key_here
EOF
```

## Running the Project
1. **Programmatic entrypoint** – import and call `run`:
   ```python
   from app import run
   response = run("Which LLM makes the best scientific agents?")
   print(response)
   ```

2. **CLI-style** – execute the sample query bundled in `app.py`:
   ```bash
   python app.py
   ```

While running, CrewAI will:
1. Route the request (supporting / exploration / both).
2. Invoke the relevant literature tools and/or SmartML modeling task.
3. Debate the perspectives and generate a final synthesized report.

## Customizing
- **Datasets:** Drop additional CSV files into `Datasets/` for SmartML. Update `JSONs/sample.json` (or craft a new spec) with filename, target column, model type, parameters, and split.
- **Agent prompts:** Tailor tone or behavior by editing the agent/task definitions in `Agents/` and `Tasks/`.
- **Publication download limits:** Modify `max_results` or `DOWNLOAD_DIR` inside the publication tools to control search breadth and storage location.

