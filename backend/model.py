from sqlalchemy import Column, Integer, String, Date, Time,DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
DATABASE_URL = "postgresql://postgres:n1m010@localhost:5432/hiTech"
# DATABASE_URL = "postgresql://alborz:n1m010@localhost:5432/hi_tech"

Base = declarative_base()

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    password = Column(String, nullable=True)
    last_name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    job = Column(String, nullable=False)
    city = Column(String, nullable=False)
    phone_number = Column(String, nullable=False)
    home_address = Column(String, nullable=True)
    education = Column(String, nullable=True)
    qr_code_path = Column(String, nullable=True)
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
    image_path = Column(String, nullable=True)
    users = relationship("UserEvent", back_populates="event")
    registration_deadline = Column(DateTime, nullable=True)
    capacity = Column(Integer, nullable=True)

class UserEvent(Base):
    __tablename__ = "user_events"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    event_id = Column(Integer, ForeignKey("events.id"), nullable=False)
    user = relationship("User", back_populates="events")
    event = relationship("Event", back_populates="users")
    attendees_count = Column(Integer, nullable=True)


class Admin(Base):
    __tablename__ = "admins"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String,unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)



Base.metadata.create_all(bind=engine)
