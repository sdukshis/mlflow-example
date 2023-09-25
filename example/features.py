import numpy as np

from sqlalchemy import String, Float, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
import pymysql


class Base(DeclarativeBase):
    pass


class Feature(Base):
    __tablename__ = "features"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(30), unique=True)
    value: Mapped[float] = mapped_column(Float)

    def __repr__(self) -> str:
        return f"{self.name}: {self.value}"


class FeatureStore:
    def __init__(self, db_connection: str):
        self._engine = create_engine(db_connection)
        Base.metadata.create_all(self._engine, checkfirst=True)
        with Session(self._engine) as session:
            market = Feature(
                name="market",
                value=123.0,
            )
            answer = Feature(
                name="answer",
                value=42.0,
            )
            holidays = Feature(
                name="holidays",
                value=5,
            )
            for feature in [market, answer, holidays]:
                if self.get(feature.name) is None:
                    session.add(feature)
            session.commit()

    def get(self, name: str) -> float:
        with Session(self._engine) as session:
            query = select(Feature).where(Feature.name == name)
            feature = session.scalar(query)
            if feature is None:
                return np.nan
            return feature.value
