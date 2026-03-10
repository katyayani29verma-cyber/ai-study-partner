"""Example usage of security modules"""
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from security.auth import AuthManager
from security.rbac import RBACManager, UserRole, Permission, require_permission
from security.validation import InputValidator, DocumentUpload
from security.encryption import FieldEncryption
from security.csrf import CSRFManager
from security.rate_limit import RateLimiter, RateLimitConfig
from security.audit import AuditLogger
from security.privacy import PrivacyManager
from security.headers import SecurityHeadersMiddleware
from security.config import settings


# Initialize security components
auth_manager = AuthManager(settings.SECRET_KEY)
csrf_manager = CSRFManager(settings.SECRET_KEY)
rate_limiter = RateLimiter()
audit_logger = AuditLogger()
privacy_manager = PrivacyManager()
field_encryption = FieldEncryption(settings.MASTER_KEY)

# Create FastAPI app
app = FastAPI(title="AI Study Partner - Secure API")

# Add security middleware
app.add_middleware(SecurityHeadersMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# Example: User Registration with Password Hashing
@app.post("/auth/register")
async def register(email: str, password: str):
    """Register new user with secure password hashing"""
    
    # Validate input
    if not InputValidator.validate_email(email):
        raise HTTPException(status_code=400, detail="Invalid email")
    
    # Hash password
    hashed_password = auth_manager.hash_password(password)
    
    # Store user in database
    # user = User(email=email, hashed_password=hashed_password)
    # db.add(user)
    # db.commit()
    
    return {"message": "User registered successfully"}


# Example: Login with JWT Token
@app.post("/auth/login")
async def login(email: str, password: str, request: Request):
    """Login with rate limiting and audit logging"""
    
    # Rate limiting
    rate_limit_key = f"login:{email}"
    if not rate_limiter.is_allowed(
        rate_limit_key,
        *RateLimitConfig.LOGIN_ATTEMPTS
    ):
        await audit_logger.log_authentication(
            user_id=email,
            success=False,
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent", ""),
            error_message="Rate limit exceeded"
        )
        raise HTTPException(status_code=429, detail="Too many login attempts")
    
    # Verify credentials
    # user = db.query(User).filter(User.email == email).first()
    # if not user or not auth_manager.verify_password(password, user.hashed_password):
    #     await audit_logger.log_authentication(...)
    #     raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Create tokens
    access_token = auth_manager.create_access_token({"sub": email})
    refresh_token = auth_manager.create_refresh_token({"sub": email})
    
    # Log successful authentication
    await audit_logger.log_authentication(
        user_id=email,
        success=True,
        ip_address=request.client.host,
        user_agent=request.headers.get("user-agent", "")
    )
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


# Example: Protected Endpoint with RBAC
@app.post("/documents/upload")
@RBACManager.require_permission(Permission.CREATE_DOCUMENT)
async def upload_document(
    document: DocumentUpload,
    request: Request,
    current_user = Depends()  # Would use actual dependency
):
    """Upload document with validation and RBAC"""
    
    # CSRF validation
    csrf_token = request.headers.get("X-CSRF-Token")
    if not csrf_token or not csrf_manager.validate_token(csrf_token, str(current_user.id)):
        raise HTTPException(status_code=403, detail="Invalid CSRF token")
    
    # Rate limiting
    rate_limit_key = f"upload:{current_user.id}"
    if not rate_limiter.is_allowed(
        rate_limit_key,
        *RateLimitConfig.API_UPLOAD
    ):
        raise HTTPException(status_code=429, detail="Upload limit exceeded")
    
    # Audit log
    await audit_logger.log_data_modification(
        user_id=str(current_user.id),
        action="CREATE",
        resource="document",
        resource_id="new_doc_id",
        ip_address=request.client.host,
        request_data=document.dict(),
        success=True
    )
    
    return {"message": "Document uploaded successfully"}


# Example: Data Export (GDPR)
@app.get("/user/data-export")
async def export_user_data(current_user = Depends()):
    """Export all user data"""
    
    export_data = await privacy_manager.export_user_data(str(current_user.id))
    
    return JSONResponse(
        content=export_data,
        headers={
            "Content-Disposition": f"attachment; filename=data_export_{current_user.id}.json"
        }
    )


# Example: Data Deletion (GDPR)
@app.delete("/user/data")
async def delete_user_data(current_user = Depends()):
    """Delete all user data"""
    
    success = await privacy_manager.delete_user_data(str(current_user.id))
    
    if success:
        return {"message": "Your data deletion request has been received"}
    else:
        raise HTTPException(status_code=500, detail="Failed to delete data")


# Example: Field Encryption
def encrypt_user_phone(phone: str) -> str:
    """Encrypt phone number before storing"""
    return field_encryption.encrypt(phone)


def decrypt_user_phone(encrypted_phone: str) -> str:
    """Decrypt phone number when retrieving"""
    return field_encryption.decrypt(encrypted_phone)


# Example: Input Validation
@app.post("/profile/update")
async def update_profile(
    name: str,
    email: str,
    current_user = Depends()
):
    """Update user profile with validation"""
    
    # Validate inputs
    if InputValidator.check_sql_injection(name):
        raise HTTPException(status_code=400, detail="Invalid name")
    
    if InputValidator.check_xss_patterns(name):
        raise HTTPException(status_code=400, detail="Invalid characters in name")
    
    if not InputValidator.validate_email(email):
        raise HTTPException(status_code=400, detail="Invalid email")
    
    # Sanitize inputs
    sanitized_name = InputValidator.sanitize_html(name)
    
    return {"message": "Profile updated successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
