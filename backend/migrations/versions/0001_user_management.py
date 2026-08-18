"""Initial user management schema.

Revision ID: 0001_user_management
Revises:
"""

from alembic import op

from backend.app import models  # noqa: F401
from backend.app.db.base import Base

revision = "0001_user_management"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    Base.metadata.create_all(bind=op.get_bind())


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind())

