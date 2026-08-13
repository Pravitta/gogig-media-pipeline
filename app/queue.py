import os
from abc import ABC, abstractmethod
from redis import Redis
from rq import Queue

class QueueService(ABC):
    @abstractmethod
    def enqueue_job(self, func_or_name, job_id, **kwargs):
        pass

class RQQueueService(QueueService):
    def __init__(self, redis_url: str):
        self.redis_conn = Redis.from_url(redis_url)
        self.queue = Queue("image_processing", connection=self.redis_conn)

    def enqueue_job(self, func_or_name, job_id, **kwargs):
        # We pass the job_id as the primary argument to the worker function,
        # and also as the kwargs 'job_id' to match RQ's internal signature if needed.
        return self.queue.enqueue(
            func_or_name,
            job_id,
            job_id=job_id,
            **kwargs
        )

class InMemoryQueueService(QueueService):
    """
    For local development when Redis is not available.
    In a real implementation, this might run tasks in a background thread.
    """
    def enqueue_job(self, func_or_name, job_id, **kwargs):
        import threading
        
        def run_task():
            if isinstance(func_or_name, str):
                import importlib
                module_name, func_name = func_or_name.rsplit('.', 1)
                module = importlib.import_module(module_name)
                func = getattr(module, func_name)
            else:
                func = func_or_name
            
            try:
                func(job_id)
            except Exception as e:
                print(f"In-memory task failed: {e}")
                
        thread = threading.Thread(target=run_task)
        thread.start()
        return None

# Factory to get the configured queue
def get_queue_service() -> QueueService:
    use_in_memory = os.getenv("USE_IN_MEMORY_QUEUE", "true").lower() == "true"
    if use_in_memory:
        return InMemoryQueueService()
    
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    try:
        r = Redis.from_url(redis_url, socket_timeout=1)
        r.ping()
        return RQQueueService(redis_url)
    except Exception:
        # Fallback to in-memory queue when Redis server is unreachable
        return InMemoryQueueService()

# Singleton instance
image_queue = get_queue_service()
