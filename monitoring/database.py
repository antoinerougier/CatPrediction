from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///monitoring/predictions.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    predicted_class = Column(String)
    confidence = Column(Float)
    prob_cat = Column(Float)
    prob_dog = Column(Float)
    feedback = Column(
        Boolean, nullable=True
    )  # None = pas encore de feedback, True = correct, False = incorrect
    true_class = Column(String, nullable=True)  # si l'utilisateur corrige


def init_db():
    Base.metadata.create_all(bind=engine)
