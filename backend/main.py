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
from sqlalchemy import desc
from model import User, SessionLocal
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
from datetime import date, time
from model import Admin,Event,User,UserEvent
from fastapi.security import OAuth2PasswordRequestForm
from itsdangerous import URLSafeTimedSerializer
from typing import List
from fastapi.encoders import jsonable_encoder
from fastapi.concurrency import run_in_threadpool
from fastapi import Form
from fastapi.security import OAuth2PasswordBearer
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.responses import JSONResponse
import os


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")




app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class EventAttendanceInput(BaseModel):
    event_id: int
    attendees_count: int


class Token(BaseModel):
    access_token: str
    token_type: str



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


class UserRegister(BaseModel):
    name: str
    last_name: str
    email: EmailStr
    job: str
    city: str
    phone_number: str
    home_address: str | None = None
    education: str | None = None
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str


class EventCreate(BaseModel):
    title: str
    description: str
    date: date
    time: time
    person_in_charge: str
    address: str
    class Config:
        from_attributes = True
class AdminLogin(BaseModel):
    name: str
    password: str


class AdminCreate(BaseModel):
    name: str
    email: str
    password: str


class AdminLoginRequest(BaseModel):
    email: str
    password: str


class UserProfile(BaseModel):
    name: str
    last_name: str
    job: str
    city: str
    phone_number: str
    home_address: str | None = None
    education: str | None = None



class EventRegistrationResponse(BaseModel):
    user_id: int
    event_id: int
    message: str
    class Config:
        orm_mode = True


class UserProfile(BaseModel):
    name: str
    last_name: str
    email: EmailStr
    job: str
    city: str
    phone_number: str
    education: str



QR_CODE_PATH = Path("static/qrcodes/")

QR_CODE_PATH.mkdir(parents=True, exist_ok=True)

EVENT_IMAGES_PATH = Path("static/event_images/")
EVENT_IMAGES_PATH.mkdir(parents=True, exist_ok=True)

SECRET_KEY = "ksdjfksjdhfksdfhksdkjhf"
SIGNING_SALT = "ksdfn,sdnfjdsjfkjsdklf"

serializer = URLSafeTimedSerializer(SECRET_KEY)


def create_access_token(data: dict, expires_delta: timedelta = timedelta(hours=24)):
    expire = (datetime.utcnow() + expires_delta).timestamp()
    data.update({"exp": expire})
    token = serializer.dumps(data, salt=SIGNING_SALT)
    return token


def verify_access_token(token: str):
    try:
        data = serializer.loads(token, salt=SIGNING_SALT, max_age=86400)
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
        token = token.split(" ")[1]
        try:
            payload = verify_access_token(token)
            admin_id = payload.get("sub")
            if not admin_id:
                raise HTTPException(status_code=403, detail="Invalid token: admin_id not found")
            admin = db.query(Admin).filter(Admin.id == int(admin_id)).first()
            if not admin:
                raise HTTPException(status_code=404, detail="Admin not found")
        except ValueError:
            raise HTTPException(status_code=403, detail="Could not validate credentials")
        request.state.admin = admin
        response = await func(request, db)
        if response is None:
            raise HTTPException(status_code=500, detail="Response was None")

        return response

    return wrapper




def generate_qr_code(user_id: int, event_id: int) -> str:
    data = f"user:{user_id}-event:{event_id}"
    qr = qrcode.make(data)
    qr_code_filename = f"{user_id}_{event_id}.png"
    qr_code_path = QR_CODE_PATH / qr_code_filename
    qr.save(qr_code_path)
    return str(qr_code_path)



def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# Routes





def save_event_image(image: UploadFile) -> str:
    # Ensure the directory exists
    os.makedirs(EVENT_IMAGES_PATH, exist_ok=True)

    # Define the file path
    image_path = EVENT_IMAGES_PATH / image.filename

    # Save the image to the specified path
    with open(image_path, "wb") as buffer:
        buffer.write(image.file.read())

    # Return the file path as a string
    return str(image_path)





@app.post("/admin/scan_qr")
def scan_qr(qr_code_data: str, db: Session = Depends(get_db)):
    try:
        user_id_str, event_id_str = qr_code_data.split("-")
        user_id = int(user_id_str.split(":")[1])
        event_id = int(event_id_str.split(":")[1])
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid QR code format")
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    db_event = db.query(Event).filter(Event.id == event_id).first()
    if not db_event:
        raise HTTPException(status_code=404, detail="Event not found")
    if db_event.id != event_id:
        raise HTTPException(status_code=403, detail="User is not registered for this event")
    return {"message": "QR code is valid and user is registered for the event"}





@app.post("/register/{event_id}")
def register_for_event(event_id: int, user_id: int, db: Session = Depends(get_db)):
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    user_event = db.query(UserEvent).filter(
        UserEvent.user_id == user_id,
        UserEvent.event_id == event_id
    ).first()
    if user_event:
        raise HTTPException(status_code=400, detail="User already registered for this event")
    new_user_event = UserEvent(user_id=user_id, event_id=event_id)
    db.add(new_user_event)
    db.commit()
    db.refresh(new_user_event)
    return {"message": "User successfully registered for the event"}






@app.get("/events", response_model=List[EventResponse])
@admin_required
async def get_all_events(request: Request, db: Session = Depends(get_db)):
    admin = request.state.admin
    events = db.query(Event).all()
    return events


from datetime import datetime

@app.post("/register_event/{event_id}", response_model=EventRegistrationResponse)
async def register_for_event(
        event_id: int,
        user_profile: UserProfile,
        db: Session = Depends(get_db)
):
    # Fetch the event
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    # Validate registration deadline
    if event.registration_deadline and datetime.now() > event.registration_deadline:
        raise HTTPException(status_code=400, detail="Registration for this event has closed")

    # Validate event capacity
    current_registration_count = db.query(UserEvent).filter(UserEvent.event_id == event_id).count()
    if event.capacity and current_registration_count >= event.capacity:
        raise HTTPException(status_code=400, detail="This event has reached its maximum capacity")

    # Check if user exists
    existing_user = db.query(User).filter(User.email == user_profile.email).first()
    if not existing_user:
        # Create a new user if not found
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

    # Generate QR code for the user-event registration
    qr_code_path = generate_qr_code(user_id, event_id)

    return EventRegistrationResponse(
        user_id=user_id,
        event_id=event_id,
        message="User successfully registered for the event",
        qr_code_path=qr_code_path
    )

@app.post("/registered_user_event_register/{event_id}", response_model=EventRegistrationResponse)
async def registered_user_event_register(
        event_id: int,
        user_email: str,
        db: Session = Depends(get_db)
):
    # Fetch the event
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    # Validate registration deadline
    if event.registration_deadline and datetime.now() > event.registration_deadline:
        raise HTTPException(status_code=400, detail="Registration for this event has closed")

    # Validate event capacity
    current_registration_count = db.query(UserEvent).filter(UserEvent.event_id == event_id).count()
    if event.capacity and current_registration_count >= event.capacity:
        raise HTTPException(status_code=400, detail="This event has reached its maximum capacity")

    # Check if the user exists
    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Ensure the user is registered to the site (must have a password)
    if not user.password:
        raise HTTPException(
            status_code=403,
            detail="User is not fully registered on the site. Please complete registration."
        )

    # Check if the user is already registered for the event
    existing_registration = db.query(UserEvent).filter(
        UserEvent.user_id == user.id,
        UserEvent.event_id == event_id
    ).first()
    if existing_registration:
        raise HTTPException(status_code=400, detail="User is already registered for this event")

    # Register the user for the event
    user_event = UserEvent(user_id=user.id, event_id=event_id)
    db.add(user_event)
    db.commit()

    # Generate QR code for the user-event registration
    qr_code_path = generate_qr_code(user.id, event_id)

    return EventRegistrationResponse(
        user_id=user.id,
        event_id=event_id,
        message="User successfully registered for the event",
        qr_code_path=qr_code_path
    )


@app.get("/api/latest-event", response_model=dict)
def get_latest_event(db: Session = Depends(get_db)):
    """
    Retrieve the event with the largest ID.
    """
    latest_event = db.query(Event).order_by(desc(Event.id)).first()
    if not latest_event:
        raise HTTPException(status_code=404, detail="No events found.")

    return {
        "id": latest_event.id,
        "title": latest_event.title,
        "description": latest_event.description,
        "date": latest_event.date,
        "time": latest_event.time,
        "person_in_charge": latest_event.person_in_charge,
        "address": latest_event.address,
        "image_path": latest_event.image_path,
    }





@app.post("/api/register", response_model=dict)
def register(user: UserRegister, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered.")

    new_user = User(
        name=user.name,
        last_name=user.last_name,
        email=user.email,
        job=user.job,
        city=user.city,
        phone_number=user.phone_number,
        home_address=user.home_address,
        education=user.education,
        password=hash_password(user.password),
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": "User registered successfully."}




@app.post("/api/login", response_model=Token)
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user or not verify_password(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    access_token = create_access_token(data={"sub": db_user.email})
    return {"access_token": access_token, "token_type": "bearer"}



def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = verify_access_token(token)
        email = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token.")
        user = db.query(User).filter(User.email == email).first()
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid token.")
        return user
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))




@app.get("/api/profile", response_model=UserProfile)
def view_profile(current_user: User = Depends(get_current_user)):
    return UserProfile(
        name=current_user.name,
        last_name=current_user.last_name,
        job=current_user.job,
        city=current_user.city,
        phone_number=current_user.phone_number,
        home_address=current_user.home_address,
        education=current_user.education,
    )




@app.put("/api/profile", response_model=UserProfile)
def edit_profile(updated_data: UserProfile, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    current_user.name = updated_data.name
    current_user.last_name = updated_data.last_name
    current_user.job = updated_data.job
    current_user.city = updated_data.city
    current_user.phone_number = updated_data.phone_number
    current_user.home_address = updated_data.home_address
    current_user.education = updated_data.education

    db.add(current_user)
    db.commit()
    db.refresh(current_user)

    return UserProfile(
        name=current_user.name,
        last_name=current_user.last_name,
        job=current_user.job,
        city=current_user.city,
        phone_number=current_user.phone_number,
        home_address=current_user.home_address,
        education=current_user.education,
    )




@app.post("/register-attendance", response_model=dict)
def register_attendance(
    data: EventAttendanceInput,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    event = db.query(Event).filter(Event.id == data.event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found.")
    existing_entry = db.query(UserEvent).filter(
        UserEvent.user_id == current_user.id,
        UserEvent.event_id == data.event_id
    ).first()
    if existing_entry:
        raise HTTPException(
            status_code=400, detail="You are already registered for this event."
        )

    new_entry = UserEvent(
        user_id=current_user.id,
        event_id=data.event_id,
        attendees_count=data.attendees_count,
    )
    db.add(new_entry)
    db.commit()
    db.refresh(new_entry)

    return {"message": "Attendance registered successfully."}
# -------------------------------------------------------------------------------------------------------------

from convertdate import persian
from datetime import datetime

def convert_jalali_to_gregorian(jalali_date):
    # jalali_date is in the format 'YYYY/MM/DD'
    year, month, day = map(int, jalali_date.split('/'))
    gregorian_date = persian.to_gregorian(year, month, day)
    return datetime(gregorian_date[0], gregorian_date[1], gregorian_date[2])


def verify_token_from_cookie(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=403, detail="Authorization token is missing")

    try:
        # Replace with actual token verification logic
        payload = verify_access_token(token)
        admin_id = payload.get("sub")

        if not admin_id:
            raise HTTPException(status_code=403, detail="Invalid token: admin_id not found")

        return admin_id
    except Exception as e:
        raise HTTPException(status_code=403, detail=f"Token validation failed: {str(e)}")


@app.post("/admin/login")
async def admin_login_page(
        request: Request,
        email: str = Form(...),
        password: str = Form(...),
        db: Session = Depends(get_db),
):
    # Query the admin based on the email
    admin = db.query(Admin).filter(Admin.email == email).first()

    if not admin or not verify_password(password, admin.hashed_password):
        # If credentials are invalid, return JSON with an error message
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid credentials"}
        )

    # Generate the access token
    access_token_expires = timedelta(hours=1)
    access_token = create_access_token(data={"sub": str(admin.id)}, expires_delta=access_token_expires)

    # Set the access token as an HTTP-only cookie
    response = RedirectResponse(url="/admin/dashboard", status_code=302)
    response.set_cookie(key="access_token", value=access_token, httponly=True)

    return response



@app.get("/admin/dashboard", response_class=HTMLResponse)
def admin_dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})



@app.post("/admin/register")
def admin_register(admin: AdminCreate, db: Session = Depends(get_db)):
    db_admin = db.query(Admin).filter(Admin.email == admin.email).first()
    if db_admin:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = hash_password(admin.password)
    new_admin = Admin(
        name=admin.name,
        email=admin.email,
        hashed_password=hashed_password,
    )
    db.add(new_admin)
    db.commit()
    db.refresh(new_admin)
    return {"message": "Admin successfully registered"}


@app.get("/admin/login", response_class=HTMLResponse)
def show_login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/admin/events", response_class=HTMLResponse)
async def admin_events(request: Request, db: Session = Depends(get_db)):
    # Get the token from cookies
    token = request.cookies.get("access_token")

    if not token:
        return RedirectResponse("/admin/login", status_code=302)

    try:
        # Verify the token (this assumes you have a function to verify the access token)
        payload = verify_access_token(token)
        admin_id = payload.get("sub")

        if not admin_id:
            raise HTTPException(status_code=403, detail="Invalid token: admin_id not found")
    except Exception as e:
        raise HTTPException(status_code=403, detail=f"Token validation failed: {str(e)}")

    # Query events from the database
    events = db.query(Event).all()

    return templates.TemplateResponse("admin_events.html", {"request": request, "events": events})
from datetime import datetime
@app.post("/admin/events", response_class=HTMLResponse)
async def create_event(
    request: Request,
    db: Session = Depends(get_db),
    title: str = Form(...),
    description: str = Form(...),
    date: str = Form(...),  # Jalali date in 'YYYY/MM/DD' format
    time: str = Form(...),
    person_in_charge: str = Form(...),
    address: str = Form(...),
    image: UploadFile = File(...),
    registration_deadline: str = Form(None),  # Optional
    capacity: int = Form(None)               # Optional
):
    # Verify the admin's token from cookies
    admin_id = verify_token_from_cookie(request)

    # Save the uploaded image
    image_path = save_event_image(image)

    # Convert the Jalali date to Gregorian before saving
    try:
        # Ensure the date is in Jalali format 'YYYY/MM/DD'
        date_in_gregorian = convert_jalali_to_gregorian(date)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid Jalali date format. Use 'YYYY/MM/DD'.")

    # Parse registration_deadline with ISO 8601 support
    deadline = None
    if registration_deadline:
        try:
            # Handle ISO 8601 format like "2024-12-12T10:56"
            deadline = datetime.fromisoformat(registration_deadline)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid registration_deadline format. Use ISO 8601.")

    # Create and save the event in the database
    db_event = Event(
        title=title,
        description=description,
        date=date_in_gregorian,  # Save the Gregorian date
        time=time,
        person_in_charge=person_in_charge,
        address=address,
        image_path=image_path,
        registration_deadline=deadline,
        capacity=capacity
    )
    db.add(db_event)
    db.commit()
    db.refresh(db_event)

    return RedirectResponse("/admin/events", status_code=302)



@app.get("/admin/events/{event_id}", response_class=HTMLResponse)
async def view_event(event_id: int, request: Request, db: Session = Depends(get_db)):
    # Verify the token from the cookie
    admin_id = verify_token_from_cookie(request)

    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    return templates.TemplateResponse("view_event.html", {"request": request, "event": event})


@app.post("/admin/events/{event_id}", response_class=HTMLResponse)
async def delete_event(event_id: int, request: Request, db: Session = Depends(get_db)):
    # Verify the token from the cookie
    admin_id = verify_token_from_cookie(request)

    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    db.delete(event)
    db.commit()

    return RedirectResponse("/admin/events", status_code=302)
@app.get("/admin/events", response_class=HTMLResponse)
def list_events(request: Request, db: Session = Depends(get_db)):
    # Fetch events from the database
    events = db.query(Event).all()
    return templates.TemplateResponse("events.html", {"request": request, "events": events})


@app.get("/admin/events/{event_id}/users", response_class=HTMLResponse)
def get_registered_users_page(event_id: int, request: Request, db: Session = Depends(get_db)):
    # Verify the token and get the admin's ID from the cookie
    admin_id = verify_token_from_cookie(request)

    # Fetch event and users
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        return templates.TemplateResponse("events.html", {"request": request, "error": "Event not found"})

    user_events = db.query(UserEvent).filter(UserEvent.event_id == event_id).all()
    users = []
    for user_event in user_events:
        user = db.query(User).filter(User.id == user_event.user_id).first()
        if user:
            users.append({
                "name": user.name,
                "last_name": user.last_name,
                "email": user.email,
                "job": user.job,
                "city": user.city,
                "phone_number": user.phone_number,
                "education": user.education,
            })

    # Return the page with users and event name
    return templates.TemplateResponse(
        "registered_users.html", {"request": request, "users": users, "event_name": event.name}
    )

@app.get("/admin/events/{event_id}/edit", response_class=HTMLResponse)
async def edit_event_page(event_id: int, request: Request, db: Session = Depends(get_db)):
    # Get the event from the database
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    # Return the edit page with the event data
    return templates.TemplateResponse("edit_event.html", {"request": request, "event": event})



@app.post("/admin/events/{event_id}/edit", response_class=HTMLResponse)
async def edit_event(
    event_id: int,
    request: Request,
    db: Session = Depends(get_db),
    title: str = Form(...),
    description: str = Form(...),
    date: str = Form(...),
    time: str = Form(...),
    person_in_charge: str = Form(...),
    address: str = Form(...),
    image: UploadFile = File(None),  # Optional
    registration_deadline: str = Form(None),
    capacity: int = Form(None)
):
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
    if image:
        event.image_path = save_event_image(image)
    if registration_deadline:
        event.registration_deadline = datetime.strptime(registration_deadline, "%Y-%m-%d %H:%M:%S")
    if capacity:
        event.capacity = capacity

    db.commit()
    db.refresh(event)

    return RedirectResponse(f"/admin/events/{event_id}", status_code=302)


from datetime import datetime



@app.get("/admin/events", response_class=HTMLResponse)
def get_event_selection_page(request: Request, db: Session = Depends(get_db)):
    # Verify token from cookies (reuse the function you created earlier)
    admin_id = verify_token_from_cookie(request)

    # Get all events from the database
    events = db.query(Event).all()

    return templates.TemplateResponse(
        "event_selection.html", {"request": request, "events": events}
    )


@app.get("/admin/event_selection", response_class=HTMLResponse)
def get_event_selection_page(request: Request, db: Session = Depends(get_db)):
    # Verify token from cookies (reuse the function you created earlier)
    admin_id = verify_token_from_cookie(request)

    # Get all events from the database
    events = db.query(Event).all()

    return templates.TemplateResponse(
        "event_selection.html", {"request": request, "events": events}
    )
