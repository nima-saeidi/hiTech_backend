"""Initial migration

Revision ID: f856354a7021
Revises: e2690487b8da
Create Date: 2024-12-14 09:12:21.560393

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f856354a7021'
down_revision: Union[str, None] = 'e2690487b8da'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('users', sa.Column('home_address', sa.String(), nullable=True))
    op.drop_column('users', 'hashed_password')
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('users', sa.Column('hashed_password', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.drop_column('users', 'home_address')
    # ### end Alembic commands ###