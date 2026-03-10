"""Audit logging for security events"""
from datetime import datetime
from typing import Optional, Dict, Any
import logging
import json


logger = logging.getLogger("audit")


class AuditLogger:
    """Log security and user actions for audit trail"""
    
    def __init__(self, db_session=None):
        """Initialize audit logger"""
        self.db = db_session
    
    async def log_action(
        self,
        user_id: str,
        action: str,
        resource: str,
        resource_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        request_data: Optional[Dict[str, Any]] = None,
        response_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log user action"""
        
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'resource_id': resource_id,
            'ip_address': ip_address,
            'user_agent': user_agent,
            'success': success,
            'error_message': error_message,
        }
        
        # Log to file
        logger.info(json.dumps(audit_entry))
        
        # Log to database if available
        if self.db:
            try:
                from sqlalchemy import Column, String, Boolean, DateTime, JSON, UUID
                from sqlalchemy.ext.declarative import declarative_base
                from uuid import uuid4
                
                # Create audit log entry
                audit_log_data = {
                    'id': str(uuid4()),
                    'user_id': user_id,
                    'action': action,
                    'resource': resource,
                    'resource_id': resource_id,
                    'ip_address': ip_address,
                    'user_agent': user_agent,
                    'success': success,
                    'error_message': error_message,
                    'request_data': request_data,
                    'response_data': response_data,
                    'timestamp': datetime.utcnow(),
                }
                
                # Insert into database
                # This would use the actual AuditLog model
                # db.add(AuditLog(**audit_log_data))
                # db.commit()
            except Exception as e:
                logger.error(f"Failed to log to database: {str(e)}")
    
    async def log_authentication(
        self,
        user_id: str,
        success: bool,
        ip_address: str,
        user_agent: str,
        error_message: Optional[str] = None
    ) -> None:
        """Log authentication attempt"""
        await self.log_action(
            user_id=user_id,
            action="LOGIN" if success else "LOGIN_FAILED",
            resource="authentication",
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message
        )
    
    async def log_data_access(
        self,
        user_id: str,
        resource: str,
        resource_id: str,
        ip_address: str,
        success: bool = True
    ) -> None:
        """Log data access"""
        await self.log_action(
            user_id=user_id,
            action="DATA_ACCESS",
            resource=resource,
            resource_id=resource_id,
            ip_address=ip_address,
            success=success
        )
    
    async def log_data_modification(
        self,
        user_id: str,
        action: str,  # CREATE, UPDATE, DELETE
        resource: str,
        resource_id: str,
        ip_address: str,
        request_data: Optional[Dict] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> None:
        """Log data modification"""
        await self.log_action(
            user_id=user_id,
            action=f"{action}_DOCUMENT",
            resource=resource,
            resource_id=resource_id,
            ip_address=ip_address,
            request_data=request_data,
            success=success,
            error_message=error_message
        )
    
    async def log_permission_denied(
        self,
        user_id: str,
        action: str,
        resource: str,
        ip_address: str
    ) -> None:
        """Log permission denied event"""
        await self.log_action(
            user_id=user_id,
            action=f"PERMISSION_DENIED_{action}",
            resource=resource,
            ip_address=ip_address,
            success=False,
            error_message="Insufficient permissions"
        )
