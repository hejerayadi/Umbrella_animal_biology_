from pathlib import Path
from dotenv import load_dotenv
import os

p = Path(__file__).resolve().parent / '.env'
print('computed_env_path', p)
print('exists', p.exists())
load_dotenv(p, override=True)
print('AZURE_FLUX_ENDPOINT', os.getenv('AZURE_FLUX_ENDPOINT'))
print('AZURE_FLUX_API_KEY', bool(os.getenv('AZURE_FLUX_API_KEY')))
