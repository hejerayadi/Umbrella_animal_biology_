from backend.agents.Protein_visualization.app.configuration.settings import Settings


def test_local_base_url_uses_configured_host_and_port() -> None:
    settings = Settings(app_host="0.0.0.0", app_port=8010)

    assert settings.local_base_url == "http://0.0.0.0:8010"
