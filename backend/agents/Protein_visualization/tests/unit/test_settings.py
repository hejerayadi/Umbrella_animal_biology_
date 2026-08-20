import re
from pathlib import Path

from backend.agents.Protein_visualization.app.configuration.settings import SERVICE_ROOT, Settings


def test_local_base_url_uses_configured_host_and_port() -> None:
    settings = Settings(app_host="0.0.0.0", app_port=8010)

    assert settings.local_base_url == "http://0.0.0.0:8010"


def test_every_documented_variable_binds_to_a_setting() -> None:
    """`.env.example` is the operator's contract, so it cannot name a dead key.

    ``extra="ignore"`` means a variable no field matches is read from `.env`,
    accepted and then silently dropped - the deployment looks configured and
    behaves as though it never was. Renaming a field without renaming the
    documented variable is exactly how that happens.
    """
    example = Path(SERVICE_ROOT, ".env.example").read_text(encoding="utf-8")
    documented = [match.group(1) for match in re.finditer(r"(?m)^([A-Z0-9_]+)=", example)]

    assert documented, ".env.example documents no variables"
    assert [name for name in documented if name.lower() not in Settings.model_fields] == []
