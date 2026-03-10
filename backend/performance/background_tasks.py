"""Background tasks for asynchronous operations"""
import logging
import asyncio
from typing import Callable, Any
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class BackgroundTask:
    """Background task wrapper"""
    
    def __init__(self, task_id: str, name: str, func: Callable, *args, **kwargs):
        self.task_id = task_id
        self.name = name
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.status = TaskStatus.PENDING
        self.created_at = datetime.utcnow()
        self.started_at = None
        self.completed_at = None
        self.result = None
        self.error = None
    
    async def execute(self) -> Any:
        """Execute the task"""
        try:
            self.status = TaskStatus.RUNNING
            self.started_at = datetime.utcnow()
            logger.info(f"Task started: {self.name} ({self.task_id})")
            
            # Execute function
            if asyncio.iscoroutinefunction(self.func):
                self.result = await self.func(*self.args, **self.kwargs)
            else:
                self.result = self.func(*self.args, **self.kwargs)
            
            self.status = TaskStatus.COMPLETED
            self.completed_at = datetime.utcnow()
            logger.info(f"Task completed: {self.name} ({self.task_id})")
            
            return self.result
        except Exception as e:
            self.status = TaskStatus.FAILED
            self.error = str(e)
            self.completed_at = datetime.utcnow()
            logger.error(f"Task failed: {self.name} ({self.task_id}): {e}")
            raise
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
        }


class TaskQueue:
    """Simple task queue for background tasks"""
    
    def __init__(self):
        self.tasks = {}
        self.queue = asyncio.Queue()
    
    async def add_task(self, task_id: str, name: str, func: Callable, *args, **kwargs) -> str:
        """Add task to queue"""
        task = BackgroundTask(task_id, name, func, *args, **kwargs)
        self.tasks[task_id] = task
        await self.queue.put(task)
        logger.info(f"Task queued: {name} ({task_id})")
        return task_id
    
    async def process_tasks(self):
        """Process tasks from queue"""
        while True:
            try:
                task = await self.queue.get()
                await task.execute()
                self.queue.task_done()
            except Exception as e:
                logger.error(f"Error processing task: {e}")
    
    def get_task(self, task_id: str) -> dict:
        """Get task status"""
        if task_id not in self.tasks:
            return None
        return self.tasks[task_id].to_dict()
    
    def get_all_tasks(self) -> list:
        """Get all tasks"""
        return [task.to_dict() for task in self.tasks.values()]


# Global task queue
task_queue = TaskQueue()


# Example background tasks
async def send_email(email: str, subject: str, body: str) -> bool:
    """Send email (example background task)"""
    logger.info(f"Sending email to {email}: {subject}")
    # Simulate email sending
    await asyncio.sleep(1)
    logger.info(f"Email sent to {email}")
    return True


async def generate_report(user_id: int, report_type: str) -> dict:
    """Generate report (example background task)"""
    logger.info(f"Generating {report_type} report for user {user_id}")
    # Simulate report generation
    await asyncio.sleep(2)
    logger.info(f"Report generated for user {user_id}")
    return {
        "user_id": user_id,
        "report_type": report_type,
        "generated_at": datetime.utcnow().isoformat(),
    }


async def sync_data(source: str, destination: str) -> bool:
    """Sync data (example background task)"""
    logger.info(f"Syncing data from {source} to {destination}")
    # Simulate data sync
    await asyncio.sleep(3)
    logger.info(f"Data synced from {source} to {destination}")
    return True
