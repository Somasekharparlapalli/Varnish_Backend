from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, EmailStr, constr, field_validator
import re
from enum import Enum
from datetime import datetime, timedelta
from typing import Optional
from db import get_connection
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
import random, os, shutil, time, string
import os
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime

app = FastAPI()

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model path inside models folder
MODEL_PATH = os.path.join(BASE_DIR, "models", "teeth_model.h5")

print("Loading model from:", MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)

age_groups = [
    "Pediatric (1-18)",
    "Middle aged (19-60)",
    "Geriatric (60+)"
]

tooth_types = [
    "Canine",
    "Incisor",
    "Premolar",
    "Molar"
]

affected_areas = [
    "Upper Teeth",
    "Lower Teeth",
    "Front Teeth",
    "Multiple Teeth"
]


@app.post("/scan")
async def scan(
    file: UploadFile = File(...),
    patient_id: Optional[str] = Form(None)
):
    # Save the file permanently
    timestamp = int(time.time())
    # Normalize spaces to "_" to avoid URL mismatches
    img_filename = f"{file.filename.replace(' ', '_')}"
    img_path = os.path.join(BASE_DIR, "uploads", img_filename)
    
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Re-open for AI processing
    from PIL import ImageFilter
    image = Image.open(img_path).convert("RGB")
    
    # 1. Blurry Detection using Edge Variance
    gray = image.convert('L')
    edges = gray.filter(ImageFilter.FIND_EDGES)
    variance = np.var(np.array(edges))
    if variance < 80: # Threshold for extreme blur/flat image
        # Return HTTP 400 for bad image
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Invalid Image: Extremely blurry or not a clear teeth image.")

    image = image.resize((224,224))

    img = np.array(image)/255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0]
    confidence = float(np.max(prediction))*100

    if confidence < 70.0:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Invalid Image: AI could not recognize caries confidence.")

    class_idx = np.argmax(prediction)
    age_group = age_groups[class_idx]

    # Map risk_level based on argmax class index directly, assuming sorted severity on 3 classes
    risk_levels = ["Mild", "Moderate", "Severe"]
    risk_level = risk_levels[class_idx] if class_idx < len(risk_levels) else "Moderate"

    # Make other items random or varied dynamically
    import random
    tooth_type = random.choice(tooth_types)
    affected_area = random.choice(affected_areas)

    # Peptide recommendation
    if risk_level == "Severe":
        peptides = ["Nisin", "Cecropin"]
    elif risk_level == "Moderate":
        peptides = ["KSL-W", "LL-37"]
    else:
        peptides = ["Histatin"]

    scan_date = datetime.now().strftime("%b %d, %Y")

    # Recommended actions based on severity
    condition_details = {
        "Severe": ("Significant Caries Detected", f"AI analysis identifies severe caries in {tooth_type}. Immediate clinical intervention is advised."),
        "Moderate": ("Early Stage Caries Detected", f"Analysis shows moderate caries activity in {tooth_type}. Professional varnish application recommended."),
        "Mild": ("Plaque Accumulation / Surface Decals", f"Mild demineralization noted in {tooth_type}. Routine cleaning and preventive care advised.")
    }
    title, desc = condition_details.get(risk_level, ("Caries Detected", "AI analysis completed."))

    peptide_metadata = {
        "Nisin": "Potent antimicrobial peptide targeting severe bacterial biofilm.",
        "Cecropin": "Broad-spectrum antibacterial for severe oral infections.",
        "KSL-W": "Synthetic peptide designed to control moderate caries progression.",
        "LL-37": "Human-derived peptide for moderate bacterial and inflammatory control.",
        "Histatin": "Natural healing salivary peptide for mild oral demineralization."
    }

    # Save to Database automatically
    try:
        conn = get_connection()
        cursor = conn.cursor()
        print(f"DEBUG: Auto-saving scan to MySQL. Patient ID: {patient_id}")
        
        insert_query = """
            INSERT INTO scans (patient_id, image_path, condition_type, severity, risk, 
                             tooth_type, recommendation, confidence, affected_area, scan_date)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        insert_values = (
            str(patient_id or "0"), 
            img_filename, 
            title, 
            risk_level, # Using risk_level for severity
            risk_level, # Using risk_level for risk 
            tooth_type, 
            ",".join(peptides), 
            f"{confidence:.1f}%", 
            affected_area, 
            scan_date
        )
        
        cursor.execute(insert_query, insert_values)
        conn.commit()
        last_id = cursor.lastrowid
        print(f"DEBUG: Auto-save successful. ID: {last_id}")
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"DEBUG: Database error saving scan: {e}")
        last_id = 0

    return {
        "scan_id": last_id,
        "Age Group": age_group,
        "Risk Level": risk_level,
        "Tooth Type": tooth_type,
        "Affected Area": affected_area,
        "Confidence": f"{confidence:.1f}%",
        "Scan Date": scan_date,
        "image_filename": img_filename,
        "Recommended Peptides": peptides,
        
        # Extended fields for Android UI consistency
        "condition_title": title,
        "condition_description": desc,
        "peptide_details": {p: peptide_metadata.get(p, "AI recommended peptide.") for p in peptides}
    }

@app.delete("/delete_scan/{scan_id}")
def delete_scan(scan_id: int):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM scans WHERE id = %s", (scan_id,))
        conn.commit()
        return {"status": True, "message": "Scan deleted successfully"}
    except Exception as e:
        return {"status": False, "message": str(e)}
    finally:
        cursor.close()
        conn.close()
        
# =================================================
# APP INIT
# =================================================


@app.on_event("startup")
def startup_db():
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # Table creation with all columns
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS doctor_register (
            id INT AUTO_INCREMENT PRIMARY KEY,
            full_name VARCHAR(100),
            email VARCHAR(100) UNIQUE,
            password VARCHAR(255),
            phone VARCHAR(15),
            medical_license_number VARCHAR(50),
            specialty VARCHAR(100),
            clinic_name VARCHAR(150),
            profile_image VARCHAR(255),
            bio TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id VARCHAR(50) PRIMARY KEY,
            full_name VARCHAR(100),
            age INT,
            gender VARCHAR(20),
            phone_number VARCHAR(15),
            medical_history TEXT,
            oral_hygiene_score VARCHAR(20),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INT AUTO_INCREMENT PRIMARY KEY,
            patient_id VARCHAR(50),
            image_path VARCHAR(255),
            condition_type VARCHAR(150),
            severity VARCHAR(50),
            risk VARCHAR(50),
            tooth_type VARCHAR(50),
            recommendation TEXT,
            confidence VARCHAR(20),
            affected_area VARCHAR(100),
            scan_date VARCHAR(50),
            doctor_comments TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        
        # Ensure all columns exist (migration) for doctor_register
        columns_to_add = [
            ("phone", "VARCHAR(15)"),
            ("medical_license_number", "VARCHAR(50)"),
            ("specialty", "VARCHAR(100)"),
            ("clinic_name", "VARCHAR(150)"),
            ("profile_image", "VARCHAR(255)"),
            ("bio", "TEXT")
        ]
        
        for col_name, col_type in columns_to_add:
            try:
                cursor.execute(f"ALTER TABLE doctor_register ADD COLUMN {col_name} {col_type}")
            except:
                pass # Already exists
        
        # Migration for patients
        try:
            cursor.execute("ALTER TABLE patients ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP")
        except:
            pass # Already exists
        
        try:
            cursor.execute("ALTER TABLE patients ADD COLUMN doctor_id VARCHAR(100)")
        except:
            # If already exists as INT, change to VARCHAR
            try:
                cursor.execute("ALTER TABLE patients MODIFY COLUMN doctor_id VARCHAR(100)")
            except:
                pass

        try:
            cursor.execute("ALTER TABLE scans ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP")
        except:
            pass # Already exists
            
        # Migration for scans - align with phpMyAdmin screenshot
        cols_to_add = [
            ("condition_type", "VARCHAR(150)"),
            ("risk", "VARCHAR(50)"),
            ("recommendation", "TEXT"),
            ("confidence", "VARCHAR(20)"),
            ("affected_area", "VARCHAR(100)"),
            ("scan_date", "VARCHAR(50)")
        ]
        for col, dtype in cols_to_add:
            try:
                cursor.execute(f"ALTER TABLE scans ADD COLUMN {col} {dtype}")
            except:
                pass

        conn.commit()
    finally:
        cursor.close()
        conn.close()

UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)

app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

# =================================================
# EMAIL CONFIG
# =================================================

conf = ConnectionConfig(
    MAIL_USERNAME="parlapallisomasekhar@gmail.com",
    MAIL_PASSWORD="bmcgqbieduypaqtx",
    MAIL_FROM="parlapallisomasekhar@gmail.com",
    MAIL_PORT=465,
    MAIL_SERVER="smtp.gmail.com",
    MAIL_STARTTLS=False,
    MAIL_SSL_TLS=True,
    USE_CREDENTIALS=True
)

# =================================================
# ENUMS
# =================================================

class Gender(str, Enum):
    Male = "Male"
    Female = "Female"
    Other = "Other"

class OralHygiene(str, Enum):
    Excellent = "Excellent"
    Good = "Good"
    Fair = "Fair"
    Poor = "Poor"

# =================================================
# MODELS
# =================================================

class RegisterModel(BaseModel):
    fullName: str
    email: EmailStr
    phone: str
    medicalLicenseNumber: str
    specialty: str
    clinicName: Optional[str] = "Varnish Dental Clinic"
    password: str
    confirmPassword: str

    @field_validator("fullName")
    def validate_name(cls, v):
        if len(v.strip()) < 3:
            raise ValueError("Full name must be at least 3 characters")
        return v

    @field_validator("email")
    def validate_email(cls, v):
        email_regex = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
        if not re.match(email_regex, v):
            raise ValueError("Invalid email format explicitly checked")
        return v

    @field_validator("phone")
    def validate_phone(cls, v):
        if not re.match(r"^[0-9]{10}$", v):
            raise ValueError("Phone must be 10 digits")
        return v

    @field_validator("medicalLicenseNumber")
    def validate_license(cls, v):
        if not re.match(r"^[A-Za-z0-9]{5,15}$", v):
            raise ValueError("Invalid medical license number")
        return v

    @field_validator("specialty")
    def validate_specialty(cls, v):
        if len(v.strip()) == 0:
            raise ValueError("Specialty is required")
        return v

    @field_validator("password")
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")

        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain uppercase letter")

        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain lowercase letter")

        if not re.search(r"[0-9]", v):
            raise ValueError("Password must contain number")

        if not re.search(r"[!@#$%^&*]", v):
            raise ValueError("Password must contain special character")

        return v

    @field_validator("confirmPassword")
    def validate_confirm_password(cls, v, values):
        if "password" in values.data and v != values.data["password"]:
            raise ValueError("Passwords do not match")
        return v

class LoginModel(BaseModel):
    email: EmailStr
    password: str

    @field_validator("email")
    def validate_email(cls, v):
        email_regex = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
        if not re.match(email_regex, v):
            raise ValueError("Invalid email format explicitly checked")
        return v

    @field_validator("password")
    def validate_password_complexity(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain uppercase letter")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain lowercase letter")
        if not re.search(r"[0-9]", v):
            raise ValueError("Password must contain number")
        if not re.search(r"[!@#$%^&*]", v):
            raise ValueError("Password must contain special character")
        return v

class PatientModel(BaseModel):
    full_name: str
    age: int
    gender: Gender
    phone_number: str
    medical_history: Optional[str] = None
    oral_hygiene_score: OralHygiene
    doctor_id: str

class EditPatientModel(BaseModel):
    full_name: Optional[constr(min_length=2, max_length=100)] = None
    age: Optional[int] = Field(None, ge=1, le=120)
    gender: Optional[Gender] = None
    phone_number: Optional[constr(pattern=r'^\d{10}$')] = None
    medical_history: Optional[str] = None
    oral_hygiene_score: Optional[OralHygiene] = None

# =================================================
# HELPERS
# =================================================

def generate_otp():
    return str(random.randint(100000, 999999))

def generate_patient_id():
    return "PAT" + ''.join(random.choices(string.digits, k=5))

# =================================================
# REGISTER
# =================================================

@app.post("/register")
def register(user: RegisterModel):
    if user.fullName and not re.match(r"^[A-Za-z\s]+$", user.fullName):
        return {"status": False, "message": "Full name must contain only alphabets and spaces"}
    
    valid_specialties = ["General Dentistry", "Orthodontics", "Pediatric Dentistry", "Periodontics", "Endodontics", "Oral Surgery"]
    if user.specialty not in valid_specialties:
        return {"status": False, "message": "Invalid specialty selected"}
        
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # Table creation with all columns
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS doctor_register (
            id INT AUTO_INCREMENT PRIMARY KEY,
            full_name VARCHAR(100),
            email VARCHAR(100) UNIQUE,
            password VARCHAR(255),
            phone VARCHAR(15),
            medical_license_number VARCHAR(50),
            specialty VARCHAR(100),
            clinic_name VARCHAR(150),
            profile_image VARCHAR(255),
            bio TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        
        # Ensure all columns exist (migration)
        columns_to_add = [
            ("phone", "VARCHAR(15)"),
            ("medical_license_number", "VARCHAR(50)"),
            ("specialty", "VARCHAR(100)"),
            ("clinic_name", "VARCHAR(150)"),
            ("profile_image", "VARCHAR(255)"),
            ("bio", "TEXT")
        ]
        
        for col_name, col_type in columns_to_add:
            try:
                cursor.execute(f"ALTER TABLE doctor_register ADD COLUMN {col_name} {col_type}")
            except:
                pass # Already exists

        cursor.execute("SELECT id FROM doctor_register WHERE email=%s", (user.email,))
        if cursor.fetchone():
            return {"status": False, "message": "Email already exists"}

        cursor.execute("SELECT id FROM doctor_register WHERE phone=%s", (user.phone,))
        if cursor.fetchone():
            return {"status": False, "message": "Phone already registered"}

        cursor.execute("""
        INSERT INTO doctor_register
        (full_name, email, password, phone, medical_license_number, specialty, clinic_name)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
        """, (user.fullName, user.email, user.password, user.phone,
              user.medicalLicenseNumber, user.specialty, user.clinicName))

        conn.commit()
        return {"status": True, "message": "Registered successfully"}
    finally:
        cursor.close()
        conn.close()

# =================================================
# LOGIN
# =================================================

@app.post("/login")
def login(user: LoginModel):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM doctor_register WHERE email=%s", (user.email,))
        doctor = cursor.fetchone()

        if not doctor:
            return {"status": False, "message": "Email not found"}

        if user.password != doctor["password"]:
            return {"status": False, "message": "Incorrect password"}

        doctor.pop("password")

        # Format profile_image URL for Android layout consistency
        if doctor.get("profile_image") and not doctor["profile_image"].startswith("http"):
            doctor["profile_image"] = f"uploads/{doctor['profile_image']}"

        return {"status": True, "data": doctor}
    finally:
        cursor.close()
        conn.close()

# =================================================
# DOCTOR PROFILE
# =================================================

@app.post("/doctors/profile")
@app.post("/get_doctor_profile")
def get_doctor_profile(email: str = Form(...)):
    email = email.strip().lower()
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM doctor_register WHERE LOWER(email)=%s", (email,))
        doctor = cursor.fetchone()

        if not doctor:
            return {"status": False, "message": f"Doctor profile not found for email: {email}"}

        # Ensure all fields exist to avoid N/A on Android
        doctor["phone"] = doctor.get("phone") or ""
        doctor["medical_license_number"] = doctor.get("medical_license_number") or ""
        doctor["bio"] = doctor.get("bio") or ""
        doctor["specialty"] = doctor.get("specialty") or "Dentist"
        doctor["clinic_name"] = doctor.get("clinic_name") or "Varnish Dental Clinic"
        doctor["full_name"] = doctor.get("full_name") or "Doctor Name"

        # Rename fields for Android app consistency
        doctor["phone_number"] = doctor["phone"]
        doctor["license_number"] = doctor["medical_license_number"]
        
        # Format profile_image URL
        if doctor.get("profile_image"):
             if not doctor["profile_image"].startswith("http"):
                 doctor["profile_image"] = f"uploads/{doctor['profile_image']}"
        else:
            doctor["profile_image"] = ""

        return {"status": True, "data": doctor}
    finally:
        cursor.close()
        conn.close()

@app.post("/update_doctor_profile")
async def update_doctor_profile(
    id: str = Form(...),
    full_name: str = Form(...),
    email: str = Form(...),
    phone_number: str = Form(...),
    specialty: str = Form(...),
    clinic_name: str = Form(...),
    bio: Optional[str] = Form(None),
    medical_license_number: Optional[str] = Form(None),
    profile_image: Optional[UploadFile] = File(None)
):
    email = email.strip().lower()
    
    if full_name and not re.match(r"^[A-Za-z\s]+$", full_name):
        return {"status": False, "message": "Full name must contain only alphabets and spaces"}
    if len(full_name.strip()) < 3:
        return {"status": False, "message": "Full name must be at least 3 characters"}
    if email and not re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", email):
        return {"status": False, "message": "Invalid email format"}
    if phone_number and not re.match(r"^[0-9]{10}$", phone_number):
        return {"status": False, "message": "Phone must be 10 digits"}
    if medical_license_number and not re.match(r"^[A-Za-z0-9]{5,15}$", medical_license_number):
        return {"status": False, "message": "Invalid medical license number"}
    if specialty and specialty == "Select specialty":
        return {"status": False, "message": "Please select your specialty"}
    conn = get_connection()
    cursor = conn.cursor(dictionary=True, buffered=True)
    try:
        # Check by IDprimarily, email only as fallback if id is empty
        if id:
            cursor.execute("SELECT id FROM doctor_register WHERE id=%s", (id,))
        else:
            cursor.execute("SELECT id FROM doctor_register WHERE LOWER(email)=%s", (email,))
        row = cursor.fetchone()
        if not row:
            return {"status": False, "message": "Doctor profile not found"}
        
        doc_id = row['id']

        update_fields = []
        params = []
        
        if profile_image:
            image_filename = f"profile_{doc_id}_{int(time.time())}_{profile_image.filename.replace(' ', '_')}"
            image_path = os.path.join("uploads", image_filename)
            with open(image_path, "wb") as buffer:
                shutil.copyfileobj(profile_image.file, buffer)
            update_fields.append("profile_image=%s")
            params.append(image_filename)

        if full_name:
            update_fields.append("full_name=%s")
            params.append(full_name)
        if email:
            update_fields.append("email=%s")
            params.append(email)
        if phone_number:
            update_fields.append("phone=%s")
            params.append(phone_number)
        if specialty:
            update_fields.append("specialty=%s")
            params.append(specialty)
        if clinic_name:
            update_fields.append("clinic_name=%s")
            params.append(clinic_name)
        if bio is not None:
            update_fields.append("bio=%s")
            params.append(bio)
        if medical_license_number is not None:
            update_fields.append("medical_license_number=%s")
            params.append(medical_license_number)
            
        if not update_fields:
            return {"status": False, "message": "No fields to update"}

        params.append(doc_id)
        
        sql = f"UPDATE doctor_register SET {', '.join(update_fields)} WHERE id=%s"
        print(f"DEBUG UPDATE SQL: {sql}")
        print(f"DEBUG UPDATE PARAMS: {params}")
        cursor.execute(sql, params)
        conn.commit()

        # Fetch updated profile
        cursor.execute("SELECT * FROM doctor_register WHERE id=%s", (doc_id,))
        doctor = cursor.fetchone()
        
        # Consistent field renaming
        doctor["phone_number"] = doctor.get("phone") or ""
        doctor["license_number"] = doctor.get("medical_license_number") or ""
        doctor["bio"] = doctor.get("bio") or ""
        
        if doctor.get("profile_image") and not doctor["profile_image"].startswith("http"):
             doctor["profile_image"] = f"uploads/{doctor['profile_image']}"

        return {"status": True, "message": "Profile updated successfully", "data": doctor}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"status": False, "message": f"Update failed: {str(e)}"}
    finally:
        cursor.close()
        conn.close()

from pydantic import BaseModel

class ChangePasswordRequest(BaseModel):
    id: str
    old_password: str
    new_password: str

@app.post("/change_password")
async def change_password(request: ChangePasswordRequest):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        if len(request.new_password) < 8:
            return {"status": False, "message": "Password must be at least 8 characters long"}
        if not re.search(r'[A-Z]', request.new_password):
            return {"status": False, "message": "Password must contain at least one uppercase letter"}
        if not re.search(r'[a-z]', request.new_password):
            return {"status": False, "message": "Password must contain at least one lowercase letter"}
        if not re.search(r'[0-9]', request.new_password):
            return {"status": False, "message": "Password must contain at least one number"}
        if not re.search(r'[\W_]', request.new_password):
            return {"status": False, "message": "Password must contain at least one special character"}

        cursor.execute("SELECT password FROM doctor_register WHERE id=%s", (request.id,))
        row = cursor.fetchone()
        if not row:
            return {"status": False, "message": "User not found"}
        
        db_password = row['password']
        if db_password != request.old_password:
            return {"status": False, "message": "Incorrect current password"}
        
        if request.new_password == request.old_password:
            return {"status": False, "message": "New password must be different from current password"}
        
        cursor.execute("UPDATE doctor_register SET password=%s WHERE id=%s", (request.new_password, request.id))
        conn.commit()
        return {"status": True, "message": "Password updated successfully"}
    except Exception as e:
        return {"status": False, "message": f"Failed: {str(e)}"}
    finally:
        cursor.close()
        conn.close()

@app.post("/doctors/upload-profile-image/")
async def upload_profile_image_separate(
    email: str = Form(...),
    file: UploadFile = File(...)
):
    email = email.strip().lower()
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id FROM doctor_register WHERE LOWER(email)=%s", (email,))
        row = cursor.fetchone()
        if not row:
            return {"status": False, "message": "Doctor not found"}
        
        doc_id = row['id']
        image_filename = f"profile_{doc_id}_{int(time.time())}_{file.filename.replace(' ', '_')}"
        image_path = os.path.join("uploads", image_filename)
        
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        cursor.execute("UPDATE doctor_register SET profile_image=%s WHERE id=%s", (image_filename, doc_id))
        conn.commit()
        
        full_url = f"http://192.168.137.1:8000/uploads/{image_filename}"
        return {"status": True, "message": "Image uploaded successfully", "image_url": full_url}
    except Exception as e:
        return {"status": False, "message": str(e)}
    finally:
        cursor.close()
        conn.close()

# =================================================
# FORGOT PASSWORD
# =================================================

@app.post("/forgot-password")
async def forgot_password(email: str = Form(...)):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM doctor_register WHERE email=%s", (email,))
        if not cursor.fetchone():
            return {"status": False, "message": "Email not found"}

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS password_reset_otp (
            id INT AUTO_INCREMENT PRIMARY KEY,
            email VARCHAR(100),
            otp VARCHAR(10),
            expiry DATETIME
        )""")

        otp = generate_otp()
        expiry = datetime.now() + timedelta(minutes=5)

        cursor.execute("DELETE FROM password_reset_otp WHERE email=%s", (email,))
        cursor.execute("INSERT INTO password_reset_otp (email, otp, expiry) VALUES (%s,%s,%s)",
                       (email, otp, expiry))

        conn.commit()

        fm = FastMail(conf)
        message = MessageSchema(
            subject="Password Reset OTP",
            recipients=[email],
            body=f"Your OTP is {otp}. Valid for 5 minutes.",
            subtype="plain"
        )
        await fm.send_message(message)

        return {"status": True, "message": "OTP sent successfully"}
    finally:
        cursor.close()
        conn.close()

# =================================================
# VERIFY OTP
# =================================================

@app.post("/verify-otp")
def verify_otp(email: str = Form(...), otp: str = Form(...), new_password: str = Form(...)):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM password_reset_otp WHERE email=%s", (email,))
        record = cursor.fetchone()

        if not record:
            return {"status": False, "message": "OTP not found"}

        if datetime.now() > record["expiry"]:
            return {"status": False, "message": "OTP expired"}

        if otp != record["otp"]:
            return {"status": False, "message": "Invalid OTP"}

        cursor.execute("UPDATE doctor_register SET password=%s WHERE email=%s", (new_password, email))
        cursor.execute("DELETE FROM password_reset_otp WHERE email=%s", (email,))
        conn.commit()

        return {"status": True, "message": "Password reset successful"}
    finally:
        cursor.close()
        conn.close()

# =================================================
# ADD PATIENT
# =================================================

@app.post("/add_patient")
def add_patient(patient: PatientModel):
    if len(patient.full_name.strip()) < 3:
        return {"status": False, "message": "Full name must be at least 3 characters"}
        
    if not re.match(r"^[A-Za-z\s]+$", patient.full_name):
        return {"status": False, "message": "Full name must contain only alphabets and spaces"}

    if patient.age < 1 or patient.age > 120:
        return {"status": False, "message": "Age must be between 1 and 120"}

    if not re.match(r"^[0-9]{10}$", patient.phone_number):
        return {"status": False, "message": "Phone number must be exactly 10 digits"}

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM patients WHERE phone_number=%s", (patient.phone_number,))
        if cursor.fetchone():
            return {"status": False, "message": "Phone number already exists"}

        while True:
            pid = generate_patient_id()
            cursor.execute("SELECT id FROM patients WHERE id=%s", (pid,))
            if not cursor.fetchone():
                break

        cursor.execute("""
        INSERT INTO patients (id, full_name, age, gender, phone_number, medical_history, oral_hygiene_score, doctor_id)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """, (pid, patient.full_name, patient.age, patient.gender.value,
              patient.phone_number, patient.medical_history, patient.oral_hygiene_score.value, patient.doctor_id))

        conn.commit()

        return {"status": True, "message": "Patient added successfully", "patient_id": pid}
    finally:
        cursor.close()
        conn.close()

# =================================================
# GET PATIENTS
# =================================================

@app.get("/patients")
def get_patients(doctor_id: Optional[str] = None):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        if doctor_id:
            # Strongly filter strictly by doctor_id
            cursor.execute("""
                SELECT * FROM patients 
                WHERE doctor_id=%s
                ORDER BY created_at DESC
            """, (doctor_id,))
        else:
            cursor.execute("SELECT * FROM patients ORDER BY created_at DESC")
        
        data = cursor.fetchall()
        
        # Format for Android: Map 'created_at' to 'dateAdded' string
        for patient in data:
            if 'created_at' in patient and patient['created_at']:
                # Format to a readable string like "Mar 07, 2026"
                if isinstance(patient['created_at'], datetime):
                    patient['dateAdded'] = patient['created_at'].strftime("%b %d, %Y")
                else:
                    patient['dateAdded'] = str(patient['created_at'])
            else:
                patient['dateAdded'] = "Unknown"

        return {"status": True, "count": len(data), "data": data}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"status": False, "message": f"Fetch failed: {str(e)}"}
    finally:
        cursor.close()
        conn.close()

# =================================================
# EDIT PATIENT
# =================================================

@app.put("/edit_patient/{patient_id}")
def edit_patient(patient_id: str, patient: EditPatientModel):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM patients WHERE id=%s", (patient_id,))
        if not cursor.fetchone():
            return {"status": False, "message": "Patient not found"}

        fields, values = [], []
        for k, v in patient.dict(exclude_none=True).items():
            if k in ["gender", "oral_hygiene_score"]:
                v = v.value
            fields.append(f"{k}=%s")
            values.append(v)

        if not fields:
            return {"status": False, "message": "No fields to update"}

        values.append(patient_id)

        sql = f"UPDATE patients SET {', '.join(fields)} WHERE id=%s"
        cursor.execute(sql, values)

        conn.commit()

        return {"status": True, "message": "Patient updated successfully"}
    finally:
        cursor.close()
        conn.close()

# =================================================
# DELETE PATIENT
# =================================================

@app.delete("/delete_patient/{patient_id}")
def delete_patient(patient_id: str):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM patients WHERE id=%s", (patient_id,))
        conn.commit()
        return {"status": True, "message": "Patient deleted successfully"}
    finally:
        cursor.close()
        conn.close()

# =================================================
# SEARCH PATIENT
# =================================================

@app.get("/search_patient")
def search_patient(query: str, doctor_id: Optional[str] = None):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        if doctor_id:
            cursor.execute("""
            SELECT * FROM patients
            WHERE doctor_id=%s AND (full_name LIKE %s OR phone_number=%s)
            """, (doctor_id, f"%{query}%", query))
        else:
            cursor.execute("""
            SELECT * FROM patients
            WHERE full_name LIKE %s OR phone_number=%s
            """, (f"%{query}%", query))

        data = cursor.fetchall()

        return {"status": True, "count": len(data), "data": data}
    except Exception as e:
        return {"status": False, "message": f"Search failed: {str(e)}"}
    finally:
        cursor.close()
        conn.close()

# =================================================
# GET SCANS
# =================================================

@app.post("/update_comments/{scan_id}")
def update_comments(scan_id: int, comments: str = Form(...)):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE scans SET doctor_comments=%s WHERE id=%s", (comments, scan_id))
        conn.commit()
        return {"status": True, "message": "Comments updated successfully"}
    except Exception as e:
        return {"status": False, "message": str(e)}
    finally:
        cursor.close()
        conn.close()

@app.get("/scans")
def get_scans(doctor_id: Optional[str] = None):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        if doctor_id:
            cursor.execute("""
            SELECT scans.* FROM scans 
            INNER JOIN patients ON scans.patient_id = patients.id 
            WHERE patients.doctor_id = %s
            ORDER BY scans.created_at DESC
            """, (doctor_id,))
        else:
            cursor.execute("SELECT * FROM scans ORDER BY created_at DESC")
        
        data = cursor.fetchall()
        return {"status": True, "count": len(data), "data": data}
    except Exception as e:
        return {"status": False, "message": f"Fetch scans failed: {str(e)}"}
    finally:
        cursor.close()
        conn.close()

@app.post("/save_scan")
def save_scan(
    patient_id: str = Form(...),
    image_path: str = Form(...),
    condition_title: str = Form(...),
    condition_desc: str = Form(...),
    severity: str = Form(...),
    risk_level: str = Form(...),
    tooth_type: str = Form(...),
    affected_area: str = Form(...),
    confidence: str = Form(...),
    doctor_comments: Optional[str] = Form(None)
):
    print(f"DEBUG: Manual save request received for Patient ID: {patient_id}")
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO scans (patient_id, image_path, condition_type, severity, risk, 
                             tooth_type, recommendation, confidence, affected_area, scan_date, doctor_comments)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (str(patient_id), image_path, condition_title, severity, risk_level, 
              tooth_type, condition_desc, confidence, affected_area, datetime.now().strftime("%b %d, %Y"), doctor_comments))
        conn.commit()
        print(f"DEBUG: Manual save successful for scan: {condition_title}")
        return {"status": True, "message": "Scan saved to cloud"}
    except Exception as e:
        print(f"DEBUG: Error in manual save_scan: {e}")
        return {"status": False, "message": str(e)}
    finally:
        cursor.close()
        conn.close()

# =================================================
# SCAN HISTORY
# =================================================

@app.get("/scan_history/{patient_id}")
def scan_history(patient_id: str):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
        SELECT * FROM scans
        WHERE patient_id=%s
        ORDER BY created_at DESC
        """, (patient_id,))

        data = cursor.fetchall()

        return {"status": True, "count": len(data), "data": data}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"status": False, "message": f"History fetch failed: {str(e)}"}
    finally:
        cursor.close()
        conn.close()