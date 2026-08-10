from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
load_dotenv(Path(__file__).resolve().parent / '.env', override=True)

sys.path.insert(0, str(ROOT))

from backend.agents.Protein_visualization.flux_client import FluxClient

print('ROOT', ROOT)
print('ENV_PATH', Path(__file__).resolve().parent / '.env')
client = FluxClient()
print('ENDPOINT', client.endpoint)
print('API_KEY_OK', bool(client.api_key))
print('MODEL', client.model)

prompt = (
    'A clean 2D scientific illustration of a protein molecule showing a helical domain, '
    'beta sheet, and a highlighted binding region, with a minimal biological style and no extra labels.'
)
try:
    result = client.generate_image(prompt)
    print('RESULT_TYPE', 'url' if isinstance(result, str) and result.startswith('http') else 'data')
    print(result[:1000])
except Exception as exc:
    print('ERROR', type(exc).__name__, exc)
