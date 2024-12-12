from sqlalchemy import Column, Integer, String, Date, Time
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
# PostgreSQL Database URL
DATABASE_URL = "postgresql://postgres:n1m010@localhost:5432/hiTech"

# Base for models
Base = declarative_base()

# Database Engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# User Model
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    job = Column(String, nullable=False)
    city = Column(String, nullable=False)
    phone_number = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    education = Column(String, nullable=True)  # New field added for education
    qr_code_path = Column(String, nullable=True)  # New field to store QR code image path
    events = relationship("UserEvent", back_populates="user")



class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(String)
    date = Column(Date)
    time = Column(Time)
    person_in_charge = Column(String)
    address = Column(String)
    image_path = Column(String, nullable=True)  # New field for event image path
    users = relationship("UserEvent", back_populates="event")


class UserEvent(Base):
    __tablename__ = "user_events"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    event_id = Column(Integer, ForeignKey("events.id"), nullable=False)

    user = relationship("User", back_populates="events")
    event = relationship("Event", back_populates="users")


class Admin(Base):
    __tablename__ = "admins"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)



Base.metadata.create_all(bind=engine)
