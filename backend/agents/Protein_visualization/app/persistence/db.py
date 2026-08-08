from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


class Database:
    def __init__(self, database_url: str) -> None:
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self.engine: Engine = create_engine(database_url, pool_pre_ping=True, connect_args=connect_args)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    def session(self) -> Session:
        return self.sessions()

    def ready(self) -> bool:
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
