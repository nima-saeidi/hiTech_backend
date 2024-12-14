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
from fastapi import Form
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")




app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

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
    image_path = EVENT_IMAGES_PATH / image.filename
    with open(image_path, "wb") as buffer:
        buffer.write(image.file.read())
    return str(image_path)




@app.post("/admin/events", response_model=List[EventResponse])
async def create_event(
    request: Request,
    db: Session = Depends(get_db),
    title: str = Form(...),
    description: str = Form(...),
    date: str = Form(...),
    time: str = Form(...),
    person_in_charge: str = Form(...),
    address: str = Form(...),
    image: UploadFile = File(...)
):
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
        image_path = save_event_image(image)
        db_event = Event(
            title=title,
            description=description,
            date=date,
            time=time,
            person_in_charge=person_in_charge,
            address=address,
            image_path=image_path
        )
        db.add(db_event)
        db.commit()
        db.refresh(db_event)
        return [EventResponse.from_orm(db_event)]
    except ValueError:
        raise HTTPException(status_code=403, detail="Could not validate credentials")



@app.get("/admin/events/{event_id}", response_model=EventResponse)
async def get_event(event_id: int, request: Request, db: Session = Depends(get_db)):
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
        event = db.query(Event).filter(Event.id == event_id).first()
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        return event
    except ValueError:
        raise HTTPException(status_code=403, detail="Could not validate credentials")



@app.delete("/admin/events/{event_id}")
async def delete_event(event_id: int, request: Request, db: Session = Depends(get_db)):
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
        event = db.query(Event).filter(Event.id == event_id).first()
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        db.delete(event)
        db.commit()
        event_check = db.query(Event).filter(Event.id == event_id).first()
        if event_check:
            raise HTTPException(status_code=500, detail="Failed to delete the event")
        return {"message": "Event deleted successfully"}
    except ValueError:
        raise HTTPException(status_code=403, detail="Could not validate credentials")


@app.get("/admin/events/{event_id}/users", response_model=list[UserProfile])
def get_registered_users(event_id: int, request: Request, db: Session = Depends(get_db)):
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
    except Exception as e:
        raise HTTPException(status_code=403, detail=f"Token validation failed: {str(e)}")
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
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




@app.post("/admin/login")
def admin_login(request: AdminLoginRequest, db: Session = Depends(get_db)):
    admin = db.query(Admin).filter(Admin.email == request.email).first()
    if not admin or not verify_password(request.password, admin.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token_expires = timedelta(hours=1)
    access_token = create_access_token(data={"sub": str(admin.id)}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}





@app.get("/events", response_model=List[EventResponse])
@admin_required
async def get_all_events(request: Request, db: Session = Depends(get_db)):
    admin = request.state.admin
    events = db.query(Event).all()
    return events


@app.post("/register_event/{event_id}", response_model=EventRegistrationResponse)
async def register_for_event(
        event_id: int,
        user_profile: UserProfile,
        db: Session = Depends(get_db)
):
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
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
    qr_code_path = generate_qr_code(user_id, event_id)
    return EventRegistrationResponse(
        user_id=user_id,
        event_id=event_id,
        message="User successfully registered for the event",
        qr_code_path=qr_code_path
    )








@app.post("/register", response_model=dict)
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




@app.post("/login", response_model=Token)
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




@app.get("/profile", response_model=UserProfile)
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




@app.put("/profile", response_model=UserProfile)
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


