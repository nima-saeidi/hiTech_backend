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
from convertdate import persian
from datetime import datetime
from fastapi import FastAPI, Request, Depends
from fastapi.responses import HTMLResponse, FileResponse
from openpyxl import Workbook
from io import BytesIO
from sqlalchemy.orm import Session







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
    class Config:
        orm_mode = True
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str
    class Config:
        orm_mode = True
        from_attributes = True


class UserCreate(BaseModel):
    name: str
    last_name: str
    email: EmailStr
    job: str
    city: str
    phone_number: str
    education: str
    password: str
    class Config:
        orm_mode = True
        from_attributes = True
class UserLogin(BaseModel):
    email: EmailStr
    password: str
    class Config:
        orm_mode = True
        from_attributes = True
class UserProfile(BaseModel):
    name: str
    last_name: str
    email: EmailStr
    job: str
    city: str
    phone_number: str
    education: str
    home_address:str
    class Config:
        orm_mode = True
        from_attributes = True

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
    qr_code_path:str
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
    home_address:str



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


def generate_qr_code(user_id: int, event_id: int, attendees_count: int) -> str:
    # Include attendees_count in the QR code data
    data = f"user:{user_id}-event:{event_id}-attendees:{attendees_count}"

    # Generate the QR code
    qr = qrcode.make(data)

    # Save the QR code to a file
    qr_code_filename = f"{user_id}_{event_id}.png"
    qr_code_path = QR_CODE_PATH / qr_code_filename
    qr.save(qr_code_path)

    # Return the QR code path as a string
    return str(qr_code_path)


def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)



def save_event_image(image: UploadFile) -> str:
    os.makedirs(EVENT_IMAGES_PATH, exist_ok=True)
    image_path = EVENT_IMAGES_PATH / image.filename
    with open(image_path, "wb") as buffer:
        buffer.write(image.file.read())
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
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    if event.registration_deadline and datetime.now() > event.registration_deadline:
        raise HTTPException(status_code=400, detail="Registration for this event has closed")
    current_registration_count = db.query(UserEvent).filter(UserEvent.event_id == event_id).count()
    if event.capacity and current_registration_count >= event.capacity:
        raise HTTPException(status_code=400, detail="This event has reached its maximum capacity")

    existing_user = db.query(User).filter(User.email == user_profile.email).first()
    if not existing_user:
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
    existing_registration = db.query(UserEvent).filter(
        UserEvent.user_id == user_id,
        UserEvent.event_id == event_id
    ).first()

    if existing_registration:
        raise HTTPException(status_code=400, detail="User is already registered for this event")
    user_event = UserEvent(user_id=user_id, event_id=event_id)
    db.add(user_event)
    db.commit()
    qr_code_path = generate_qr_code(user_id, event_id,0)
    print(qr_code_path)
    return EventRegistrationResponse(
        user_id=user_id,
        event_id=event_id,
        qr_code_path=qr_code_path,
        message="User successfully registered for the event",
    )


@app.post("/registered_user_event_register/{event_id}", response_model=EventRegistrationResponse)
async def registered_user_event_register(
        event_id: int,
        user_email: str,
        attendees_count: int,  # Added parameter for attendees count
        db: Session = Depends(get_db)
):
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    # if event.registration_deadline and datetime.now() > event.registration_deadline:
    #     raise HTTPException(status_code=400, detail="Registration for this event has closed")

    current_registration_count = db.query(UserEvent).filter(UserEvent.event_id == event_id).count()
    if event.capacity and current_registration_count >= event.capacity:
        raise HTTPException(status_code=400, detail="This event has reached its maximum capacity")

    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not user.password:
        raise HTTPException(
            status_code=403,
            detail="User is not fully registered on the site. Please complete registration."
        )

    existing_registration = db.query(UserEvent).filter(
        UserEvent.user_id == user.id,
        UserEvent.event_id == event_id
    ).first()
    if existing_registration:
        raise HTTPException(status_code=400, detail="User is already registered for this event")

    # Restrict attendees count to a maximum of 5
    if attendees_count > 5:
        raise HTTPException(status_code=400, detail="You can bring a maximum of 5 attendees.")

    # Register the user for the event
    user_event = UserEvent(user_id=user.id, event_id=event_id, attendees_count=attendees_count)
    db.add(user_event)
    db.commit()

    # Generate QR code with attendees count
    qr_code_path = generate_qr_code(user.id, event_id, attendees_count)
    return EventRegistrationResponse(
        user_id=user.id,
        event_id=event_id,
        message="User successfully registered for the event",
        qr_code_path=qr_code_path
    )


@app.get("/api/latest-event", response_model=dict)
def get_latest_event(db: Session = Depends(get_db)):
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
        email=current_user.email,  # Include the email field
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
        email=current_user.email

    )




@app.post("/register-attendance", response_model=dict)
def register_attendance(
    data: EventAttendanceInput,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Check if the event exists
    event = db.query(Event).filter(Event.id == data.event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found.")

    # Check if the user is already registered for this event
    existing_entry = db.query(UserEvent).filter(
        UserEvent.user_id == current_user.id,
        UserEvent.event_id == data.event_id
    ).first()
    if existing_entry:
        raise HTTPException(
            status_code=400, detail="You are already registered for this event."
        )

    # Validate the number of attendees
    attendees_count = data.attendees_count
    if attendees_count <= 0:
        raise HTTPException(
            status_code=400, detail="Attendees count cannot be negative."
        )
    if attendees_count >= 5:
        raise HTTPException(
            status_code=400, detail="You can bring a maximum of 5 attendees."
        )

    # Check event capacity if applicable
    if event.capacity is not None:
        # Calculate total attendees including the user
        total_attendees = attendees_count + 1
        registered_count = db.query(UserEvent).filter(
            UserEvent.event_id == data.event_id
        ).count()

        if registered_count + total_attendees > event.capacity:
            raise HTTPException(
                status_code=400, detail="Event capacity exceeded."
            )

    # Register the user for the event
    new_entry = UserEvent(
        user_id=current_user.id,
        event_id=data.event_id,
        attendees_count=attendees_count,
    )
    db.add(new_entry)
    db.commit()
    db.refresh(new_entry)

    # Generate the QR code
    qr_code_path = generate_qr_code(
        user_id=current_user.id,
        event_id=data.event_id,
        attendees_count=attendees_count
    )

    return {
        "message": "Attendance registered successfully.",
        "qr_code_path": qr_code_path
    }





# -------------------------------------------------------------------------------------------------------------



def convert_jalali_to_gregorian(jalali_date):
    year, month, day = map(int, jalali_date.split('/'))
    gregorian_date = persian.to_gregorian(year, month, day)
    return datetime(gregorian_date[0], gregorian_date[1], gregorian_date[2])


def verify_token_from_cookie(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=403, detail="Authorization token is missing")

    try:
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
    admin = db.query(Admin).filter(Admin.email == email).first()

    if not admin or not verify_password(password, admin.hashed_password):
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid credentials"}
        )

    access_token_expires = timedelta(hours=1)
    access_token = create_access_token(data={"sub": str(admin.id)}, expires_delta=access_token_expires)
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
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse("/admin/login", status_code=302)
    try:
        payload = verify_access_token(token)
        admin_id = payload.get("sub")
        if not admin_id:
            raise HTTPException(status_code=403, detail="Invalid token: admin_id not found")
    except Exception as e:
        raise HTTPException(status_code=403, detail=f"Token validation failed: {str(e)}")
    events = db.query(Event).all()

    return templates.TemplateResponse("admin_events.html", {"request": request, "events": events})

@app.post("/admin/events", response_class=HTMLResponse)
async def create_event(
    request: Request,
    db: Session = Depends(get_db),
    title: str = Form(...),
    description: str = Form(...),
    date: str = Form(...),
    time: str = Form(...),
    person_in_charge: str = Form(...),
    address: str = Form(...),
    image: UploadFile = File(...),
    registration_deadline: str = Form(None),
    capacity: int = Form(None)
):
    admin_id = verify_token_from_cookie(request)
    image_path = save_event_image(image)

    try:
        date_in_gregorian = convert_jalali_to_gregorian(date)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid Jalali date format. Use 'YYYY/MM/DD'.")

    deadline = None
    if registration_deadline:
        try:
            deadline = datetime.fromisoformat(registration_deadline)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid registration_deadline format. Use ISO 8601.")
    db_event = Event(
        title=title,
        description=description,
        date=date_in_gregorian,
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
    admin_id = verify_token_from_cookie(request)
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return templates.TemplateResponse("view_event.html", {"request": request, "event": event})


@app.post("/admin/events/{event_id}", response_class=HTMLResponse)
async def delete_event(event_id: int, request: Request, db: Session = Depends(get_db)):
    admin_id = verify_token_from_cookie(request)
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    # Delete related rows in user_events
    db.query(UserEvent).filter(UserEvent.event_id == event_id).delete()

    # Delete the event
    db.delete(event)
    db.commit()

    return RedirectResponse("/admin/events", status_code=302)


@app.get("/admin/events", response_class=HTMLResponse)
def list_events(request: Request, db: Session = Depends(get_db)):
    events = db.query(Event).all()
    return templates.TemplateResponse("events.html", {"request": request, "events": events})




@app.get("/admin/events/{event_id}/users", response_class=HTMLResponse)
def get_registered_users_page(event_id: int, request: Request, db: Session = Depends(get_db)):
    admin_id = verify_token_from_cookie(request)

    # Fetch the event details
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        return templates.TemplateResponse("events.html", {"request": request, "error": "Event not found"})

    # Fetch all user-event mappings for the event
    user_events = db.query(UserEvent).filter(UserEvent.event_id == event_id).all()

    # Calculate total registrations and build the user data
    total_registered_users = len(user_events)
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
                "attendees_count": user_event.attendees_count or 0,  # Default to 0 if None
            })

    # Render the template with the required data
    return templates.TemplateResponse(
        "registered_users.html",
        {
            "request": request,
            "users": users,
            "event_name": event.title,
            "event_id": event.id,
            "total_registered_users": total_registered_users,
        }
    )



from fastapi.responses import StreamingResponse


from urllib.parse import quote

@app.get("/admin/events/{event_id}/users/export", response_class=StreamingResponse)
def export_registered_users(event_id: int, db: Session = Depends(get_db)):
    # Fetch event details
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    # Fetch user-event mappings for the event
    user_events = db.query(UserEvent).filter(UserEvent.event_id == event_id).all()

    # Prepare user data with attendance count
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
                "attendees_count": user_event.attendees_count or 0,  # Default to 0 if None
            })

    # Create an Excel workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Registered Users"

    # Define headers including attendance count
    headers = [
        "Name", "Last Name", "Email", "Job",
        "City", "Phone Number", "Education", "Attendance Count"
    ]
    ws.append(headers)

    # Add user data to the spreadsheet
    for user in users:
        ws.append([
            user["name"],
            user["last_name"],
            user["email"],
            user["job"],
            user["city"],
            user["phone_number"],
            user["education"],
            user["attendees_count"],
        ])

    # Save the workbook to a BytesIO stream
    file = BytesIO()
    wb.save(file)
    file.seek(0)

    # Encode the filename to be safe for HTTP headers
    filename = quote(f"registered_users_{event.title}.xlsx")

    # Return the file as a streaming response
    return StreamingResponse(
        file,
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
@app.get("/admin/events/{event_id}/edit", response_class=HTMLResponse)
async def edit_event_page(event_id: int, request: Request, db: Session = Depends(get_db)):
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return templates.TemplateResponse("edit_event.html", {"request": request, "event": event})


from persiantools.jdatetime import JalaliDate
from fastapi import HTTPException


def persian_to_ascii(persian_str: str) -> str:
    persian_digits = '۰۱۲۳۴۵۶۷۸۹'
    ascii_digits = '0123456789'
    translation_table = str.maketrans(persian_digits, ascii_digits)
    return persian_str.translate(translation_table)


def jalali_to_gregorian(jalali_date: str):

    try:
        jalali_date_ascii = persian_to_ascii(jalali_date)
        jalali_parts = jalali_date_ascii.split('/')
        if len(jalali_parts) != 3:
            raise ValueError("Incorrect date format")
        jalali_year, jalali_month, jalali_day = map(int, jalali_parts)
        gregorian_date = JalaliDate(jalali_year, jalali_month, jalali_day).to_gregorian()
        return gregorian_date.strftime("%Y-%m-%d")  # Return Gregorian date in 'YYYY-MM-DD'
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Jalali date format: {jalali_date}")


@app.post("/admin/events/{event_id}/edit")
async def edit_event(
        event_id: int,
        db: Session = Depends(get_db),
        title: str = Form(...),
        description: str = Form(...),
        date: str = Form(...),
        time: str = Form(...),
        capacity: int = Form(None)
):
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    try:
        gregorian_date = jalali_to_gregorian(date)
    except HTTPException as e:
        raise e
    event.title = title
    event.description = description
    event.date = gregorian_date
    event.time = time
    if capacity:
        event.capacity = capacity
    db.commit()
    db.refresh(event)

    return RedirectResponse(f"/admin/events/{event_id}", status_code=302)


@app.get("/admin/events", response_class=HTMLResponse)
def get_event_selection_page(request: Request, db: Session = Depends(get_db)):
    admin_id = verify_token_from_cookie(request)
    events = db.query(Event).all()
    return templates.TemplateResponse(
        "event_selection.html", {"request": request, "events": events}
    )


@app.get("/admin/event_selection", response_class=HTMLResponse)
def get_event_selection_page(request: Request, db: Session = Depends(get_db)):
    admin_id = verify_token_from_cookie(request)
    events = db.query(Event).all()
    event_user_data = {}
    for event in events:
        user_events = db.query(UserEvent).filter(UserEvent.event_id == event.id).all()
        users = [user_event.user for user_event in user_events]
        event_user_data[event] = users
    return templates.TemplateResponse(
        "event_selection.html", {"request": request, "event_user_data": event_user_data}
    )


@app.get("/admin/admin_event_selection", response_class=HTMLResponse)
def event_selection_page(request: Request, db: Session = Depends(get_db)):
    events = db.query(Event).all()
    return templates.TemplateResponse("admin_event_selection.html", {"request": request, "events": events})


