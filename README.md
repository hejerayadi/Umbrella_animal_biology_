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

5. To let the frontend talk to the orchestrator, run the HTTP API instead (also from the repository root):

   ```powershell
   uvicorn backend.api:app --reload --port 8000
   ```

   This starts a server at `http://localhost:8000` with one endpoint, `POST /api/chat`, that runs a user's message through the full Planner -> Worker -> Capability Resolver loop and returns the result as JSON.

## Running the Frontend

Requirements: Node.js 18+

```powershell
cd frontend
npm install
npm run dev
```

Then open the local URL printed in the terminal.

The frontend calls the backend API at `http://localhost:8000` by default. If you're running the API on a different host/port, set `VITE_ORCHESTRATOR_API_URL` in a `.env` file inside `frontend/` before starting the dev server:

```
VITE_ORCHESTRATOR_API_URL=http://localhost:8000
```

## Running Both Together

Two terminals, both started from the repository root:

```powershell
# Terminal 1 - backend API
.\venv\Scripts\Activate.ps1
uvicorn backend.api:app --reload --port 8000

# Terminal 2 - frontend
cd frontend
npm run dev
```

Sending a chat message in the UI now runs the real orchestrator: the Agent Thinking panel reflects the actual `Planner -> Worker -> Capability Resolver` steps returned by the backend, not a simulated timeline. If the backend isn't reachable, the chat shows an error message instead of hanging silently.
