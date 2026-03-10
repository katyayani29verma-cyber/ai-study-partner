"""Comprehensive test suite - Cross-platform compatible"""
import sys
sys.path.insert(0, '.')

from security import AuthManager, RBACManager, InputValidator, EncryptionManager, UserRole, Permission
from database import create_db_engine, Base, User, StudyMaterial, init_db_session
from sqlalchemy.orm import Session


def test_auth():
    """Test authentication"""
    print("\n=== Testing Authentication ===")
    auth = AuthManager()
    
    # Test password hashing
    password = "TestPassword123"
    hashed = auth.hash_password(password)
    assert auth.verify_password(password, hashed), "Password verification failed"
    print("[OK] Password hashing and verification works")
    
    # Test token creation
    token = auth.create_access_token({"sub": "user123"})
    assert token, "Token creation failed"
    print("[OK] Token creation works")
    
    # Test token decoding
    payload = auth.decode_token(token)
    assert payload and payload.get("sub") == "user123", "Token decoding failed"
    print("[OK] Token decoding works")


def test_rbac():
    """Test RBAC"""
    print("\n=== Testing RBAC ===")
    
    # Test admin permissions
    assert RBACManager.has_permission(UserRole.ADMIN, Permission.DELETE), "Admin should have delete permission"
    print("[OK] Admin has all permissions")
    
    # Test student permissions
    assert RBACManager.has_permission(UserRole.STUDENT, Permission.READ), "Student should have read permission"
    assert not RBACManager.has_permission(UserRole.STUDENT, Permission.DELETE), "Student should not have delete permission"
    print("[OK] Student has correct permissions")
    
    # Test guest permissions
    assert RBACManager.has_permission(UserRole.GUEST, Permission.READ), "Guest should have read permission"
    assert not RBACManager.has_permission(UserRole.GUEST, Permission.WRITE), "Guest should not have write permission"
    print("[OK] Guest has correct permissions")


def test_validation():
    """Test input validation"""
    print("\n=== Testing Validation ===")
    
    # Test email validation
    assert InputValidator.validate_email("test@example.com"), "Valid email should pass"
    assert not InputValidator.validate_email("invalid-email"), "Invalid email should fail"
    print("[OK] Email validation works")
    
    # Test password validation
    assert InputValidator.validate_password("ValidPass123"), "Valid password should pass"
    assert not InputValidator.validate_password("weak"), "Weak password should fail"
    print("[OK] Password validation works")
    
    # Test sanitization
    sanitized = InputValidator.sanitize_input("<script>alert('xss')</script>")
    assert "<" not in sanitized, "Sanitization should remove dangerous characters"
    print("[OK] Input sanitization works")
    
    # Test SQL injection detection
    assert InputValidator.check_sql_injection("DROP TABLE users"), "SQL injection should be detected"
    assert not InputValidator.check_sql_injection("normal text"), "Normal text should not be flagged"
    print("[OK] SQL injection detection works")


def test_encryption():
    """Test encryption"""
    print("\n=== Testing Encryption ===")
    
    enc = EncryptionManager()
    
    # Test encryption and decryption
    original = "sensitive data"
    encrypted = enc.encrypt(original)
    assert encrypted != original, "Encrypted data should differ from original"
    print("[OK] Encryption works")
    
    decrypted = enc.decrypt(encrypted)
    assert decrypted == original, "Decrypted data should match original"
    print("[OK] Decryption works")


def test_database():
    """Test database"""
    print("\n=== Testing Database ===")
    
    # Create engine
    engine = create_db_engine()
    print("[OK] Database engine created")
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    print("[OK] Database tables created")
    
    # Initialize session
    init_db_session(engine)
    print("[OK] Database session initialized")
    
    # Check tables
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    assert len(tables) > 0, "No tables created"
    print(f"[OK] {len(tables)} tables created successfully")
    
    # List tables
    for table in sorted(tables):
        print(f"   - {table}")


def test_api():
    """Test API"""
    print("\n=== Testing API ===")
    
    from api.main import app
    from fastapi.testclient import TestClient
    
    client = TestClient(app)
    
    # Test health endpoint
    response = client.get("/health")
    assert response.status_code == 200, "Health check failed"
    assert response.json()["status"] == "healthy", "Health status incorrect"
    print("[OK] Health endpoint works")
    
    # Test root endpoint
    response = client.get("/")
    assert response.status_code == 200, "Root endpoint failed"
    print("[OK] Root endpoint works")


if __name__ == "__main__":
    print("=" * 50)
    print("COMPREHENSIVE TEST SUITE")
    print("=" * 50)
    
    try:
        test_auth()
        test_rbac()
        test_validation()
        test_encryption()
        test_database()
        test_api()
        
        print("\n" + "=" * 50)
        print("[OK] ALL TESTS PASSED!")
        print("=" * 50)
    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
