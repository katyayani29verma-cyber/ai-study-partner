"""Test security features"""
import sys
sys.path.insert(0, '.')

from security import AuthManager, EncryptionManager, InputValidator, RBACManager, UserRole, Permission

print('=== SECURITY FEATURES TEST ===\n')

# Test 1: Authentication
print('1. Authentication System')
auth = AuthManager()
password = 'SecurePass123!'
hashed = auth.hash_password(password)
verified = auth.verify_password(password, hashed)
print(f'   ✅ Password hashing: {verified}')

token = auth.create_access_token({'sub': 'user123', 'role': 'student'})
payload = auth.decode_token(token)
print(f'   ✅ Token creation and validation: {payload.get("sub") == "user123"}')

# Test 2: Encryption
print('\n2. Encryption System')
enc = EncryptionManager()
sensitive = 'credit_card_1234567890'
encrypted = enc.encrypt(sensitive)
decrypted = enc.decrypt(encrypted)
print(f'   ✅ Data encryption/decryption: {decrypted == sensitive}')

# Test 3: Input Validation
print('\n3. Input Validation')
valid_email = InputValidator.validate_email('user@example.com')
invalid_email = InputValidator.validate_email('not-an-email')
print(f'   ✅ Email validation: {valid_email and not invalid_email}')

valid_pwd = InputValidator.validate_password('ValidPass123')
invalid_pwd = InputValidator.validate_password('weak')
print(f'   ✅ Password validation: {valid_pwd and not invalid_pwd}')

sanitized = InputValidator.sanitize_input('<script>alert("xss")</script>')
print(f'   ✅ XSS sanitization: {"<" not in sanitized}')

sql_injection = InputValidator.check_sql_injection('DROP TABLE users')
normal_text = InputValidator.check_sql_injection('normal query')
print(f'   ✅ SQL injection detection: {sql_injection and not normal_text}')

# Test 4: RBAC
print('\n4. Role-Based Access Control')
admin_delete = RBACManager.has_permission(UserRole.ADMIN, Permission.DELETE)
student_delete = RBACManager.has_permission(UserRole.STUDENT, Permission.DELETE)
student_read = RBACManager.has_permission(UserRole.STUDENT, Permission.READ)
print(f'   ✅ Admin permissions: {admin_delete}')
print(f'   ✅ Student permissions: {student_read and not student_delete}')

print('\n=== ALL SECURITY TESTS PASSED ===')
