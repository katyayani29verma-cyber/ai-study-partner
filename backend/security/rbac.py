"""Role-Based Access Control - Clean implementation"""
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class UserRole(str, Enum):
    """User roles"""
    ADMIN = "admin"
    TEACHER = "teacher"
    STUDENT = "student"
    GUEST = "guest"


class Permission(str, Enum):
    """Permissions"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    MANAGE_USERS = "manage_users"
    MANAGE_CONTENT = "manage_content"


class RBACManager:
    """Role-Based Access Control manager"""
    
    # Role to permissions mapping
    ROLE_PERMISSIONS = {
        UserRole.ADMIN: [
            Permission.READ,
            Permission.WRITE,
            Permission.DELETE,
            Permission.MANAGE_USERS,
            Permission.MANAGE_CONTENT,
        ],
        UserRole.TEACHER: [
            Permission.READ,
            Permission.WRITE,
            Permission.MANAGE_CONTENT,
        ],
        UserRole.STUDENT: [
            Permission.READ,
            Permission.WRITE,
        ],
        UserRole.GUEST: [
            Permission.READ,
        ],
    }
    
    @staticmethod
    def get_permissions(role: UserRole) -> list:
        """Get permissions for a role"""
        return RBACManager.ROLE_PERMISSIONS.get(role, [])
    
    @staticmethod
    def has_permission(role: UserRole, permission: Permission) -> bool:
        """Check if role has permission"""
        permissions = RBACManager.get_permissions(role)
        return permission in permissions
    
    @staticmethod
    def check_access(role: UserRole, required_permission: Permission) -> bool:
        """Check if user has required permission"""
        return RBACManager.has_permission(role, required_permission)
