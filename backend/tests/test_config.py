from backend.app.core.config import Settings


def test_supabase_database_url_uses_psycopg_and_ssl() -> None:
    settings = Settings(
        _env_file=None,
        database_url=(
            "postgresql://postgres.project:secret%40value@"
            "aws-0-eu-north-1.pooler.supabase.com:5432/postgres"
        ),
    )

    assert settings.database_url == (
        "postgresql+psycopg://postgres.project:secret%40value@"
        "aws-0-eu-north-1.pooler.supabase.com:5432/postgres?sslmode=require"
    )


def test_local_psycopg_database_url_is_unchanged() -> None:
    database_url = "postgresql+psycopg://umbrella:secret@localhost:5432/umbrella"

    settings = Settings(_env_file=None, database_url=database_url)

    assert settings.database_url == database_url
