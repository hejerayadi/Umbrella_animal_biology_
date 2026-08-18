"""Preserve invitations when their issuing administrator is deleted.

Revision ID: 0002_nullable_invitation_issuer
Revises: 0001_user_management
"""

from alembic import op

revision = "0002_nullable_invitation_issuer"
down_revision = "0001_user_management"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("invitations_invited_by_id_fkey", "invitations", type_="foreignkey")
    op.alter_column("invitations", "invited_by_id", nullable=True)
    op.create_foreign_key(
        "invitations_invited_by_id_fkey",
        "invitations",
        "users",
        ["invited_by_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("invitations_invited_by_id_fkey", "invitations", type_="foreignkey")
    op.execute("DELETE FROM invitations WHERE invited_by_id IS NULL")
    op.alter_column("invitations", "invited_by_id", nullable=False)
    op.create_foreign_key(
        "invitations_invited_by_id_fkey",
        "invitations",
        "users",
        ["invited_by_id"],
        ["id"],
        ondelete="RESTRICT",
    )
