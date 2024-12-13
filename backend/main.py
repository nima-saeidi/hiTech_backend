from fastapi import FastAPI, HTTPException, Depends,Request
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from typing import Callable
from functools import wraps
import qrcode

from backend.model import User, SessionLocal
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
from datetime import date, time
from model import Admin,Event,User,UserEvent
from fastapi.security import OAuth2PasswordRequestForm
from itsdangerous import URLSafeTimedSerializer
from typing import List
from fastapi.encoders import jsonable_encoder
from fastapi.concurrency import run_in_threadpool






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


class EventResponse(BaseModel):
    id: int
    title: str
    description: str
    date: date
    time: time
    person_in_charge: str
    address: str
    image_path: str

    class Config:
        orm_mode = True
        from_attributes = True

class EventCreate(BaseModel):
    title: str
    description: str
    date: date  # ISO format (YYYY-MM-DD)
    time: time  # ISO format (HH:MM:SS)
    person_in_charge: str
    address: str
    class Config:
        from_attributes = True  # Enable ORM mode
class AdminLogin(BaseModel):
    name: str
    password: str


class AdminCreate(BaseModel):
    name: str
    email: str
    password: str




# Path where QR codes will be stored
QR_CODE_PATH = Path("static/qrcodes/")

# Ensure the directory exists
QR_CODE_PATH.mkdir(parents=True, exist_ok=True)

EVENT_IMAGES_PATH = Path("static/event_images/")
EVENT_IMAGES_PATH.mkdir(parents=True, exist_ok=True)

SECRET_KEY = "your_secret_key"  # Replace with your actual secret key
SIGNING_SALT = "your_signing_salt"  # Add a salt for extra security

serializer = URLSafeTimedSerializer(SECRET_KEY)


def create_access_token(data: dict, expires_delta: timedelta = timedelta(hours=24)):
    # Add expiration time to the payload
    expire = (datetime.utcnow() + expires_delta).timestamp()
    data.update({"exp": expire})

    # Create the token
    token = serializer.dumps(data, salt=SIGNING_SALT)
    return token


def verify_access_token(token: str):
    try:
        data = serializer.loads(token, salt=SIGNING_SALT, max_age=86400)  # max_age is in seconds
        return data
    except Exception as e:
        raise ValueError(f"Invalid or expired token: {str(e)}")



def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

def admin_required(func):
    @wraps(func)
    async def wrapper(request: Request, db: Session = Depends(get_db)):
        token = request.headers.get("Authorization")
        if not token or not token.startswith("Bearer "):
            raise HTTPException(status_code=403, detail="Authorization token is missing or invalid")

        token = token.split(" ")[1]  # Extract the token part

        try:
            # Assuming verify_access_token is already defined elsewhere
            payload = verify_access_token(token)
            admin_id = payload.get("sub")
            if not admin_id:
                raise HTTPException(status_code=403, detail="Invalid token: admin_id not found")

            admin = db.query(Admin).filter(Admin.id == int(admin_id)).first()
            if not admin:
                raise HTTPException(status_code=404, detail="Admin not found")
        except ValueError:
            raise HTTPException(status_code=403, detail="Could not validate credentials")

        # Set the admin in the request state
        request.state.admin = admin

        # Call the actual endpoint function with the correct arguments
        response = await func(request, db)

        # Ensure the response is valid (if expected to be a list, check here)
        if response is None:
            raise HTTPException(status_code=500, detail="Response was None")

        return response

    return wrapper




def generate_qr_code(user_id: int, event_id: int) -> str:
    data = f"user:{user_id}-event:{event_id}"  # Unique data for user and event
    print(f"Generating QR code with data: {data}")  # Debugging line to check data

    # Generate the QR code
    qr = qrcode.make(data)

    # Save the QR code image to the specified path
    qr_code_filename = f"{user_id}_{event_id}.png"
    qr_code_path = QR_CODE_PATH / qr_code_filename
    print(f"Saving QR code to: {qr_code_path}")  # Debugging line to check the path

    qr.save(qr_code_path)

    # Return the path of the saved QR code
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
    # Define the image path
    image_path = EVENT_IMAGES_PATH / image.filename

    # Save the image directly
    with open(image_path, "wb") as buffer:
        buffer.write(image.file.read())

    return str(image_path)



# Event creation API (with image upload)
from fastapi import Form
@app.post("/admin/events", response_model=List[EventResponse])
async def create_event(
    request: Request,
    db: Session = Depends(get_db),
    title: str = Form(...),
    description: str = Form(...),
    date: str = Form(...),  # Accept date as a string
    time: str = Form(...),  # Accept time as a string
    person_in_charge: str = Form(...),
    address: str = Form(...),
    image: UploadFile = File(...)
):
    # Manually check if the admin is logged in
    token = request.headers.get("Authorization")
    if not token or not token.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Authorization token is missing or invalid")

    token = token.split(" ")[1]  # Extract the token part

    try:
        # Assuming verify_access_token is already defined elsewhere
        payload = verify_access_token(token)
        admin_id = payload.get("sub")
        if not admin_id:
            raise HTTPException(status_code=403, detail="Invalid token: admin_id not found")

        admin = db.query(Admin).filter(Admin.id == int(admin_id)).first()
        if not admin:
            raise HTTPException(status_code=404, detail="Admin not found")

        # If admin is authenticated, proceed with the event creation logic
        # Save the image
        image_path = save_event_image(image)

        # Create the event
        db_event = Event(
            title=title,
            description=description,
            date=date,
            time=time,
            person_in_charge=person_in_charge,
            address=address,
            image_path=image_path
        )

        # Add to the database
        db.add(db_event)
        db.commit()
        db.refresh(db_event)

        # Return a list with the created event
        return [EventResponse.from_orm(db_event)]

    except ValueError:
        raise HTTPException(status_code=403, detail="Could not validate credentials")



@app.get("/admin/events/{event_id}", response_model=EventResponse)
async def get_event(event_id: int, request: Request, db: Session = Depends(get_db)):
    # Manually check if the admin is logged in
    token = request.headers.get("Authorization")
    if not token or not token.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Authorization token is missing or invalid")

    token = token.split(" ")[1]  # Extract the token part

    try:
        # Assuming verify_access_token is already defined elsewhere
        payload = verify_access_token(token)
        admin_id = payload.get("sub")
        if not admin_id:
            raise HTTPException(status_code=403, detail="Invalid token: admin_id not found")

        admin = db.query(Admin).filter(Admin.id == int(admin_id)).first()
        if not admin:
            raise HTTPException(status_code=404, detail="Admin not found")

        # If admin is authenticated, proceed with the logic
        event = db.query(Event).filter(Event.id == event_id).first()

        if not event:
            raise HTTPException(status_code=404, detail="Event not found")

        return event
    except ValueError:
        raise HTTPException(status_code=403, detail="Could not validate credentials")



@app.delete("/admin/events/{event_id}")
async def delete_event(event_id: int, request: Request, db: Session = Depends(get_db)):
    # Manually check if the admin is logged in
    token = request.headers.get("Authorization")
    if not token or not token.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Authorization token is missing or invalid")

    token = token.split(" ")[1]  # Extract the token part

    try:
        # Assuming verify_access_token is already defined elsewhere
        payload = verify_access_token(token)
        admin_id = payload.get("sub")
        if not admin_id:
            raise HTTPException(status_code=403, detail="Invalid token: admin_id not found")

        admin = db.query(Admin).filter(Admin.id == int(admin_id)).first()
        if not admin:
            raise HTTPException(status_code=404, detail="Admin not found")

        # If admin is authenticated, proceed with the event deletion logic
        event = db.query(Event).filter(Event.id == event_id).first()

        if not event:
            raise HTTPException(status_code=404, detail="Event not found")

        db.delete(event)
        db.commit()

        # Verify if the event was deleted
        event_check = db.query(Event).filter(Event.id == event_id).first()
        if event_check:
            raise HTTPException(status_code=500, detail="Failed to delete the event")

        return {"message": "Event deleted successfully"}
    except ValueError:
        raise HTTPException(status_code=403, detail="Could not validate credentials")


@app.put("/admin/events/{event_id}", response_model=EventResponse)
async def edit_event(
    event_id: int,
    request: Request,
    db: Session = Depends(get_db),
    title: str = Form(...),
    description: str = Form(...),
    date: str = Form(...),  # Accept date as a string
    time: str = Form(...),  # Accept time as a string
    person_in_charge: str = Form(...),
    address: str = Form(...),
    image: UploadFile = File(None)  # Image is optional for editing
):
    # Manually check if the admin is logged in
    token = request.headers.get("Authorization")
    if not token or not token.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Authorization token is missing or invalid")

    token = token.split(" ")[1]  # Extract the token part

    try:
        # Assuming verify_access_token is already defined elsewhere
        payload = verify_access_token(token)
        admin_id = payload.get("sub")
        if not admin_id:
            raise HTTPException(status_code=403, detail="Invalid token: admin_id not found")

        admin = db.query(Admin).filter(Admin.id == int(admin_id)).first()
        if not admin:
            raise HTTPException(status_code=404, detail="Admin not found")

        # If admin is authenticated, proceed with event editing logic
        event = db.query(Event).filter(Event.id == event_id).first()
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")

        # Update event fields
        event.title = title
        event.description = description
        event.date = date
        event.time = time
        event.person_in_charge = person_in_charge
        event.address = address

        # If a new image is uploaded, save it
        if image:
            image_path = save_event_image(image)
            event.image_path = image_path

        # Commit the changes to the database
        db.commit()
        db.refresh(event)

        # Return the updated event
        return EventResponse.from_orm(event)

    except ValueError:
        raise HTTPException(status_code=403, detail="Could not validate credentials")


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
def get_registered_users(event_id: int ,request: Request, db: Session = Depends(get_db)):
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


class AdminLoginRequest(BaseModel):
    email: str
    password: str

@app.post("/admin/login")
def admin_login(request: AdminLoginRequest, db: Session = Depends(get_db)):
    # Query admin by email
    admin = db.query(Admin).filter(Admin.email == request.email).first()
    if not admin or not verify_password(request.password, admin.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Generate JWT token for the admin
    access_token_expires = timedelta(hours=1)
    access_token = create_access_token(data={"sub": str(admin.id)}, expires_delta=access_token_expires)

    return {"access_token": access_token, "token_type": "bearer"}





@app.get("/events", response_model=List[EventResponse])
@admin_required
async def get_all_events(request: Request, db: Session = Depends(get_db)):
    """
    Fetch all events. Only accessible by admins.
    """
    admin = request.state.admin
    events = db.query(Event).all()
    return events







# Pydantic model for the response
class EventRegistrationResponse(BaseModel):
    user_id: int
    event_id: int
    message: str

    class Config:
        orm_mode = True


# UserProfile Model
class UserProfile(BaseModel):
    name: str
    last_name: str
    email: EmailStr
    job: str
    city: str
    phone_number: str
    education: str


# API to register a user to an event
@app.post("/register_event/{event_id}", response_model=EventRegistrationResponse)
async def register_for_event(
        event_id: int,
        user_profile: UserProfile,  # Now accepting user profile data
        db: Session = Depends(get_db)
):
    # Check if the event exists
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    # Check if the user already exists with the provided email (to avoid duplicate registrations)
    existing_user = db.query(User).filter(User.email == user_profile.email).first()

    if not existing_user:
        # If the user does not exist, create a new user
        new_user = User(
            name=user_profile.name,
            last_name=user_profile.last_name,
            email=user_profile.email,
            job=user_profile.job,
            city=user_profile.city,
            phone_number=user_profile.phone_number,
            education=user_profile.education
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        user_id = new_user.id
    else:
        user_id = existing_user.id

    # Check if the user is already registered for the event
    existing_registration = db.query(UserEvent).filter(
        UserEvent.user_id == user_id,
        UserEvent.event_id == event_id
    ).first()

    if existing_registration:
        raise HTTPException(status_code=400, detail="User is already registered for this event")

    # Register the user for the event
    user_event = UserEvent(user_id=user_id, event_id=event_id)
    db.add(user_event)
    db.commit()

    # Generate QR code for the user
    qr_code_path = generate_qr_code(user_id, event_id)

    # Return a successful response with the QR code path
    return EventRegistrationResponse(
        user_id=user_id,
        event_id=event_id,
        message="User successfully registered for the event",
        qr_code_path=qr_code_path  # Include the QR code path in the response
    )
