from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from typing import Callable
from functools import wraps
from backend.model import User, SessionLocal
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware

from fastapi.security import OAuth2PasswordRequestForm






app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific domains (e.g., ["https://example.com"]) for security.
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods.
    allow_headers=["*"],  # Allow all headers.
)

app.mount("/static", StaticFiles(directory="static"), name="static")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class UserCreate(BaseModel):
    name: str
    last_name: str
    email: EmailStr
    job: str
    city: str
    phone_number: str
    education: str
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserProfile(BaseModel):
    name: str
    last_name: str
    email: EmailStr
    job: str
    city: str
    phone_number: str
    education: str


class EventCreate(BaseModel):
    title: str
    description: str
    date: str  # ISO format (YYYY-MM-DD)
    time: str  # ISO format (HH:MM:SS)
    person_in_charge: str
    address: str

class AdminCreate(BaseModel):
    name: str
    email: str
    password: str

class EventResponse(EventCreate):
    id: int

    class Config:
        orm_mode = True


# Path where QR codes will be stored
QR_CODE_PATH = Path("static/qrcodes/")

# Ensure the directory exists
QR_CODE_PATH.mkdir(parents=True, exist_ok=True)

EVENT_IMAGES_PATH = Path("static/event_images/")
EVENT_IMAGES_PATH.mkdir(parents=True, exist_ok=True)


def admin_required(func: Callable):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Get the request and db session
        request: Request = kwargs.get("request")
        db: Session = kwargs.get("db")

        # Extract the token from the Authorization header
        token = request.headers.get("Authorization")
        if not token:
            raise HTTPException(status_code=403, detail="Authorization token is missing")

        token = token.split(" ")[1]  # Strip "Bearer " part

        # Verify and decode the token
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            admin_id = payload.get("sub")
            if admin_id is None:
                raise HTTPException(status_code=403, detail="Could not validate credentials")

            # Retrieve the admin from the database
            admin = db.query(Admin).filter(Admin.id == admin_id).first()
            if admin is None:
                raise HTTPException(status_code=404, detail="Admin not found")
        except JWTError:
            raise HTTPException(status_code=403, detail="Could not validate credentials")

        # Call the original function
        return await func(*args, **kwargs)

    return wrapper

def generate_qr_code(user_id: int, event_id: int) -> str:
    data = f"user:{user_id}-event:{event_id}"  # Unique data for user and event
    qr = qrcode.make(data)

    # Save the QR code image to the specified path
    qr_code_filename = f"{user_id}_{event_id}.png"
    qr_code_path = QR_CODE_PATH / qr_code_filename
    qr.save(qr_code_path)

    return str(qr_code_path)
# Utility functions
def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def create_user(db: Session, user: UserCreate):
    hashed_password = pwd_context.hash(user.password)
    db_user = User(
        name=user.name,
        last_name=user.last_name,
        email=user.email,
        job=user.job,
        city=user.city,
        phone_number=user.phone_number,
        education=user.education,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# Routes
@app.post("/register", response_model=UserProfile)
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user_by_email(db, user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return create_user(db, user)

@app.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = get_user_by_email(db, user.email)
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid email or password")
    return {"message": "Login successful", "user_id": db_user.id}

@app.get("/profile/{user_id}", response_model=UserProfile)
def profile(user_id: int, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserProfile(
        name=db_user.name,
        last_name=db_user.last_name,
        email=db_user.email,
        job=db_user.job,
        city=db_user.city,
        phone_number=db_user.phone_number,
        education=db_user.education
    )


def save_event_image(image: UploadFile) -> str:
    # Generate a unique filename for the image
    image_filename = f"{Path(image.filename).stem}_{int(time.time())}{Path(image.filename).suffix}"
    image_path = EVENT_IMAGES_PATH / image_filename

    # Save the image
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    return str(image_path)


# Event creation API (with image upload)
@app.post("/admin/events", response_model=EventResponse)
@admin_required
async def create_event(
        event: EventCreate,
        image: UploadFile = File(...),  # Accepting the image file
        db: Session = Depends(get_db)
):
    # Save the image
    image_path = save_event_image(image)

    # Create the event
    db_event = Event(
        title=event.title,
        description=event.description,
        date=event.date,
        time=event.time,
        person_in_charge=event.person_in_charge,
        address=event.address,
        image_path=image_path  # Store the image path in the event record
    )

    # Add to the database
    db.add(db_event)
    db.commit()
    db.refresh(db_event)

    return db_event

@app.get("/admin/events", response_model=list[EventResponse])
@admin_required
def list_events(db: Session = Depends(get_db)):
    events = db.query(Event).all()
    return events

@app.get("/admin/events/{event_id}", response_model=EventResponse)
@admin_required
def get_event(event_id: int, db: Session = Depends(get_db)):
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return event

@app.delete("/admin/events/{event_id}")
def delete_event(event_id: int, db: Session = Depends(get_db)):
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    db.delete(event)
    db.commit()
    return {"message": "Event deleted successfully"}


@app.post("/event/register", response_model=UserProfile)
def register_event(user: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user_by_email(db, user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create the user
    db_user = create_user(db, user)

    # Generate QR Code for the user registration in an event
    event_id = user.event_id  # Assuming event_id is provided in the request
    qr_code_path = generate_qr_code(db_user.id, event_id)

    # Update the user record with the generated QR code path
    db_user.qr_code_path = qr_code_path
    db.commit()
    db.refresh(db_user)

    return db_user


# Admin Scanning QR Code Endpoint
@app.post("/admin/scan_qr")
@admin_required
def scan_qr(qr_code_data: str, db: Session = Depends(get_db)):
    # Decode the QR code data (assuming it is in the form of "user:{user_id}-event:{event_id}")
    try:
        user_id_str, event_id_str = qr_code_data.split("-")
        user_id = int(user_id_str.split(":")[1])
        event_id = int(event_id_str.split(":")[1])
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid QR code format")

    # Retrieve the user and event data from the database
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    db_event = db.query(Event).filter(Event.id == event_id).first()
    if not db_event:
        raise HTTPException(status_code=404, detail="Event not found")

    # Verify that the user is registered for the event (this depends on your specific logic)
    # For example, you may have a many-to-many relationship between User and Event
    # Check that the user is indeed registered for the event.
    if db_event.id != event_id:
        raise HTTPException(status_code=403, detail="User is not registered for this event")

    return {"message": "QR code is valid and user is registered for the event"}


@app.get("/admin/events/{event_id}/users", response_model=list[UserProfile])
@admin_required
def get_registered_users(event_id: int, db: Session = Depends(get_db)):
    # Query to get the users registered for the event
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    # Get the users who are registered for this event
    user_events = db.query(UserEvent).filter(UserEvent.event_id == event_id).all()
    registered_users = []
    for user_event in user_events:
        user = db.query(User).filter(User.id == user_event.user_id).first()
        if user:
            registered_users.append(UserProfile(
                name=user.name,
                last_name=user.last_name,
                email=user.email,
                job=user.job,
                city=user.city,
                phone_number=user.phone_number,
                education=user.education
            ))

    return registered_users


@app.post("/register/{event_id}")
def register_for_event(event_id: int, user_id: int, db: Session = Depends(get_db)):
    # Check if the event exists
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    # Check if the user already registered for the event
    user_event = db.query(UserEvent).filter(
        UserEvent.user_id == user_id,
        UserEvent.event_id == event_id
    ).first()
    if user_event:
        raise HTTPException(status_code=400, detail="User already registered for this event")

    # Register the user for the event
    new_user_event = UserEvent(user_id=user_id, event_id=event_id)
    db.add(new_user_event)
    db.commit()
    db.refresh(new_user_event)

    return {"message": "User successfully registered for the event"}


@app.post("/admin/register")
def admin_register(admin: AdminCreate, db: Session = Depends(get_db)):
    db_admin = db.query(Admin).filter(Admin.email == admin.email).first()
    if db_admin:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = hash_password(admin.password)  # Define a function to hash passwords

    new_admin = Admin(
        name=admin.name,
        email=admin.email,
        hashed_password=hashed_password,
    )

    db.add(new_admin)
    db.commit()
    db.refresh(new_admin)

    return {"message": "Admin successfully registered"}


@app.post("/admin/login")
def admin_login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    admin = db.query(Admin).filter(Admin.email == form_data.username).first()
    if not admin or not verify_password(form_data.password, admin.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Generate JWT token for the admin
    access_token_expires = timedelta(hours=1)
    access_token = create_access_token(data={"sub": str(admin.id)}, expires_delta=access_token_expires)

    return {"access_token": access_token, "token_type": "bearer"}

def create_access_token(data: dict, expires_delta: timedelta = timedelta(hours=24)):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt