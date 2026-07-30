# Umbrella_animal_biology_
AI-powered multi-agent platform for animal genomics and biodiversity research. It integrates genomic databases, LLMs, and bioinformatics to analyze genomes, reconstruct missing DNA regions, discover gene-trait relationships, compare species, and explore biological knowledge through a conversational interface.

## Project Structure

- `backend/` - Python multi-agent orchestrator (LangGraph + LangChain + Azure OpenAI)
- `frontend/` - React / TanStack Start web UI

## Running the Backend

Requirements: Python 3.11+

1. From the repository root, create and activate a virtual environment:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Provide Azure OpenAI credentials in `backend/orchestrator/.env` (this file is git-ignored, create it yourself):

   ```
   azure_endpoint=https://<your-resource>.openai.azure.com/
   openai_key_azure=<your-api-key>
   ```

4. Run the orchestrator demo from the **repository root** (not from inside `backend/`), using `-m` so Python recognizes it as a package:

   ```powershell
   python -m backend.main
   ```

   Running `python backend\main.py` or `cd backend; python main.py` will fail with
   `ImportError: attempted relative import with no known parent package` - always use the `-m` form above instead.

## Running the Frontend

Requirements: Node.js 18+

```powershell
cd frontend
npm install
npm run dev
```

Then open the local URL printed in the terminal. The frontend currently runs on local mock data and is not yet wired up to the backend.
