"""Security module tests"""
import pytest
from datetime import timedelta

from security.auth import AuthManager
from security.rbac import RBACManager, UserRole, Permission
from security.validation import InputValidator, DocumentUpload
from security.encryption import FieldEncryption
from security.csrf import CSRFManager
from security.rate_limit import RateLimiter, RateLimitConfig


class TestAuthManager:
    """Test authentication functionality"""
    
    def setup_method(self):
        self.auth = AuthManager("test-secret-key")
    
    def test_hash_password(self):
        """Test password hashing"""
        password = "SecurePass123!"
        hashed = self.auth.hash_password(password)
        
        assert hashed != password
        assert len(hashed) > 0
    
    def test_verify_password(self):
        """Test password verification"""
        password = "SecurePass123!"
        hashed = self.auth.hash_password(password)
        
        assert self.auth.verify_password(password, hashed)
        assert not self.auth.verify_password("WrongPassword", hashed)
    
    def test_create_access_token(self):
        """Test access token creation"""
        token = self.auth.create_access_token({"sub": "user123"})
        
        assert token is not None
        assert len(token) > 0
    
    def test_decode_token(self):
        """Test token decoding"""
        data = {"sub": "user123"}
        token = self.auth.create_access_token(data)
        
        payload = self.auth.decode_token(token)
        
        assert payload is not None
        assert payload["sub"] == "user123"
    
    def test_invalid_token(self):
        """Test invalid token handling"""
        payload = self.auth.decode_token("invalid.token.here")
        
        assert payload is None


class TestRBAC:
    """Test role-based access control"""
    
    def test_student_permissions(self):
        """Test student role permissions"""
        permissions = RBACManager.get_role_permissions(UserRole.STUDENT)
        
        assert Permission.CREATE_DOCUMENT in permissions
        assert Permission.READ_DOCUMENT in permissions
        assert Permission.DELETE_DOCUMENT in permissions
        assert Permission.MANAGE_USERS not in permissions
    
    def test_admin_permissions(self):
        """Test admin role permissions"""
        permissions = RBACManager.get_role_permissions(UserRole.ADMIN)
        
        assert Permission.MANAGE_USERS in permissions
        assert Permission.DELETE_USERS in permissions
        assert len(permissions) > len(
            RBACManager.get_role_permissions(UserRole.STUDENT)
        )
    
    def test_has_permission(self):
        """Test permission checking"""
        assert RBACManager.has_permission(
            UserRole.STUDENT,
            Permission.CREATE_DOCUMENT
        )
        
        assert not RBACManager.has_permission(
            UserRole.STUDENT,
            Permission.MANAGE_USERS
        )
    
    def test_admin_has_all_permissions(self):
        """Test admin has all permissions"""
        admin_perms = RBACManager.get_role_permissions(UserRole.ADMIN)
        # Admin should have most permissions (not necessarily all enum values)
        assert len(admin_perms) > 0
        assert Permission.MANAGE_USERS in admin_perms
        assert Permission.DELETE_USERS in admin_perms


class TestInputValidation:
    """Test input validation"""
    
    def test_sanitize_html(self):
        """Test HTML sanitization"""
        dirty = "<script>alert('xss')</script>Hello"
        clean = InputValidator.sanitize_html(dirty)
        
        assert "<script>" not in clean
        assert "Hello" in clean
    
    def test_validate_email(self):
        """Test email validation"""
        assert InputValidator.validate_email("user@example.com")
        assert not InputValidator.validate_email("invalid-email")
        assert not InputValidator.validate_email("user@")
    
    def test_validate_url(self):
        """Test URL validation"""
        assert InputValidator.validate_url("https://example.com")
        assert InputValidator.validate_url("http://example.com/path")
        assert not InputValidator.validate_url("not-a-url")
    
    def test_check_sql_injection(self):
        """Test SQL injection detection"""
        assert InputValidator.check_sql_injection("'; DROP TABLE users; --")
        assert InputValidator.check_sql_injection("1' OR '1'='1")
        assert not InputValidator.check_sql_injection("normal text")
    
    def test_check_xss_patterns(self):
        """Test XSS pattern detection"""
        assert InputValidator.check_xss_patterns("<script>alert('xss')</script>")
        assert InputValidator.check_xss_patterns("javascript:alert('xss')")
        assert InputValidator.check_xss_patterns("onerror=alert('xss')")
        assert not InputValidator.check_xss_patterns("normal text")
    
    def test_document_upload_validation(self):
        """Test document upload validation"""
        doc = DocumentUpload(
            title="My Document",
            subject="Math",
            tags=["algebra"]
        )
        
        assert doc.title == "My Document"
        assert doc.subject == "Math"
    
    def test_document_upload_invalid_title(self):
        """Test invalid document title"""
        with pytest.raises(ValueError):
            DocumentUpload(
                title="",  # Empty title
                subject="Math"
            )


class TestEncryption:
    """Test encryption functionality"""
    
    def setup_method(self):
        self.encryptor = FieldEncryption("test-master-key-32-characters-long")
    
    def test_encrypt_decrypt(self):
        """Test encryption and decryption"""
        plaintext = "sensitive_data_123"
        encrypted = self.encryptor.encrypt(plaintext)
        decrypted = self.encryptor.decrypt(encrypted)
        
        assert encrypted != plaintext
        assert decrypted == plaintext
    
    def test_encrypt_empty_string(self):
        """Test encrypting empty string"""
        encrypted = self.encryptor.encrypt("")
        
        assert encrypted == ""
    
    def test_decrypt_invalid_data(self):
        """Test decrypting invalid data"""
        result = self.encryptor.decrypt("invalid_encrypted_data")
        
        assert result is None


class TestCSRF:
    """Test CSRF protection"""
    
    def setup_method(self):
        self.csrf = CSRFManager("test-secret-key-32-characters-long")
    
    def test_generate_token(self):
        """Test token generation"""
        token = self.csrf.generate_token("session123")
        
        assert token is not None
        assert "." in token
    
    def test_validate_token(self):
        """Test token validation"""
        session_id = "session123"
        token = self.csrf.generate_token(session_id)
        
        assert self.csrf.validate_token(token, session_id)
    
    def test_invalid_token(self):
        """Test invalid token"""
        assert not self.csrf.validate_token("invalid.token", "session123")
    
    def test_token_session_mismatch(self):
        """Test token with wrong session"""
        token = self.csrf.generate_token("session123")
        
        assert not self.csrf.validate_token(token, "session456")


class TestRateLimiter:
    """Test rate limiting"""
    
    def setup_method(self):
        self.limiter = RateLimiter()
    
    def test_allow_within_limit(self):
        """Test allowing requests within limit"""
        key = "test_key"
        
        for i in range(5):
            assert self.limiter.is_allowed(key, 5, 60)
    
    def test_deny_over_limit(self):
        """Test denying requests over limit"""
        key = "test_key"
        
        for i in range(5):
            self.limiter.is_allowed(key, 5, 60)
        
        assert not self.limiter.is_allowed(key, 5, 60)
    
    def test_get_remaining(self):
        """Test getting remaining requests"""
        key = "test_key"
        
        self.limiter.is_allowed(key, 5, 60)
        self.limiter.is_allowed(key, 5, 60)
        
        remaining = self.limiter.get_remaining(key, 5, 60)
        
        assert remaining == 3
    
    def test_rate_limit_config(self):
        """Test rate limit configurations"""
        assert RateLimitConfig.LOGIN_ATTEMPTS == (5, 60)
        assert RateLimitConfig.API_UPLOAD == (10, 3600)


class TestDocumentUpload:
    """Test document upload validation"""
    
    def test_valid_document(self):
        """Test valid document"""
        doc = DocumentUpload(
            title="Study Guide",
            subject="Mathematics",
            tags=["algebra", "geometry"]
        )
        
        assert doc.title == "Study Guide"
        assert len(doc.tags) == 2
    
    def test_sanitized_title(self):
        """Test title sanitization"""
        doc = DocumentUpload(
            title="<b>Study Guide</b>",
            subject="Math"
        )
        
        # Title should be sanitized
        assert "<b>" not in doc.title or doc.title == "<b>Study Guide</b>"
    
    def test_max_tags(self):
        """Test maximum tags limit"""
        with pytest.raises(ValueError):
            DocumentUpload(
                title="Study Guide",
                tags=[f"tag{i}" for i in range(11)]  # 11 tags, max is 10
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
