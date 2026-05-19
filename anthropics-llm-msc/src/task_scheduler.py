# src/task_scheduler.py

import asyncio
import time
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
import heapq
import threading
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class ScheduledTask:
    """Represent a scheduled task with priority and timing information."""
    task_id: str
    model: str
    func: Callable
    args: tuple
    kwargs: dict
    priority: int = 0
    scheduled_time: float = field(default_factory=time.time)
    attempts: int = 0
    max_attempts: int = 3
    
    def __lt__(self, other):
        # For heapq (min-heap): lower values = higher priority
        if self.scheduled_time != other.scheduled_time:
            return self.scheduled_time < other.scheduled_time
        return self.priority < other.priority

class SmartTaskScheduler:
    """Intelligent task scheduler that balances load across providers."""
    
    def __init__(self, rate_limits: Dict[str, Any]):
        self.rate_limits = rate_limits
        self.task_queue = []  # Priority queue using heapq
        self.model_stats = {}  # Track performance per model
        self.running = True
        self.queue_lock = threading.Lock()
        self.executor_lock = threading.Lock()
        
        # Initialize model statistics
        for model in rate_limits.keys():
            self.model_stats[model] = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'average_latency': 0.0,
                'last_request_time': 0.0,
                'consecutive_failures': 0,
                'backoff_until': 0.0
            }
        
        # Start the scheduler thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
    
    def schedule_task(self, task: ScheduledTask):
        """Add a task to the scheduling queue."""
        with self.queue_lock:
            heapq.heappush(self.task_queue, task)
    
    def schedule_batch(self, tasks: List[ScheduledTask]):
        """Schedule multiple tasks efficiently."""
        with self.queue_lock:
            for task in tasks:
                heapq.heappush(self.task_queue, task)
    
    def _run_scheduler(self):
        """Main scheduler loop that processes tasks."""
        while self.running:
            try:
                # Check if we have tasks to process
                with self.queue_lock:
                    if not self.task_queue:
                        time.sleep(0.1)
                        continue
                    
                    # Get the next task (without removing it yet)
                    next_task = self.task_queue[0]
                    current_time = time.time()
                    
                    # Check if it's time to execute this task
                    if current_time < next_task.scheduled_time:
                        time.sleep(min(0.1, next_task.scheduled_time - current_time))
                        continue
                    
                    # Check if the model is in backoff
                    model_stats = self.model_stats[next_task.model]
                    if current_time < model_stats['backoff_until']:
                        # Reschedule for after backoff
                        next_task.scheduled_time = model_stats['backoff_until']
                        heapq.heapify(self.task_queue)  # Reheapify after modification
                        continue
                    
                    # Remove the task from queue and execute it
                    task = heapq.heappop(self.task_queue)
                
                # Execute the task (outside the lock)
                self._execute_task(task)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(1)  # Prevent tight error loop
    
    def _execute_task(self, task: ScheduledTask):
        """Execute a single task and update statistics."""
        start_time = time.time()
        model = task.model
        
        try:
            # Execute the task
            result = task.func(*task.args, **task.kwargs)
            
            # Update statistics on success
            latency = time.time() - start_time
            self._update_stats_success(model, latency)
            
            return result
            
        except Exception as e:
            # Update statistics on failure
            self._update_stats_failure(model)
            
            # Reschedule if we haven't exceeded max attempts
            if task.attempts < task.max_attempts:
                task.attempts += 1
                task.scheduled_time = time.time() + self._calculate_backoff(model, task.attempts)
                
                with self.queue_lock:
                    heapq.heappush(self.task_queue, task)
                
                logger.warning(f"Task failed, rescheduling (attempt {task.attempts}/{task.max_attempts}): {e}")
            else:
                logger.error(f"Task failed after {task.max_attempts} attempts: {e}")
                raise
    
    def _update_stats_success(self, model: str, latency: float):
        """Update model statistics after a successful request."""
        stats = self.model_stats[model]
        
        stats['total_requests'] += 1
        stats['successful_requests'] += 1
        stats['last_request_time'] = time.time()
        stats['consecutive_failures'] = 0
        
        # Update running average of latency
        if stats['total_requests'] > 1:
            stats['average_latency'] = (
                stats['average_latency'] * (stats['total_requests'] - 1) + latency
            ) / stats['total_requests']
        else:
            stats['average_latency'] = latency
    
    def _update_stats_failure(self, model: str):
        """Update model statistics after a failed request."""
        stats = self.model_stats[model]
        
        stats['total_requests'] += 1
        stats['failed_requests'] += 1
        stats['consecutive_failures'] += 1
        stats['last_request_time'] = time.time()
        
        # Apply backoff for consecutive failures
        if stats['consecutive_failures'] >= 3:
            backoff_time = min(60 * (2 ** (stats['consecutive_failures'] - 3)), 300)
            stats['backoff_until'] = time.time() + backoff_time
            logger.warning(f"Model {model} in backoff for {backoff_time}s after {stats['consecutive_failures']} failures")
    
    def _calculate_backoff(self, model: str, attempt: int) -> float:
        """Calculate backoff time for a failed request."""
        base_delay = 1.0
        rate_limit = self.rate_limits.get(model)
        
        if rate_limit:
            # Use the provider's retry delay if available
            base_delay = rate_limit.retry_delay
        
        # Exponential backoff with jitter
        import random
        delay = base_delay * (2 ** (attempt - 1))
        jitter = random.uniform(0.5, 1.5)
        
        return min(delay * jitter, 300)  # Cap at 5 minutes
    
    def get_model_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all models."""
        return {
            model: {
                'success_rate': (
                    stats['successful_requests'] / stats['total_requests'] 
                    if stats['total_requests'] > 0 else 0
                ),
                'average_latency': stats['average_latency'],
                'total_requests': stats['total_requests'],
                'consecutive_failures': stats['consecutive_failures'],
                'is_in_backoff': time.time() < stats['backoff_until']
            }
            for model, stats in self.model_stats.items()
        }
    
    def optimize_task_distribution(self, available_tasks: List[ScheduledTask]) -> List[ScheduledTask]:
        """Optimize the distribution of tasks across models for better throughput."""
        # Group tasks by model
        tasks_by_model = {}
        for task in available_tasks:
            if task.model not in tasks_by_model:
                tasks_by_model[task.model] = []
            tasks_by_model[task.model].append(task)
        
        # Calculate optimal timing for each model
        current_time = time.time()
        optimized_tasks = []
        
        for model, tasks in tasks_by_model.items():
            rate_limit = self.rate_limits.get(model)
            if not rate_limit:
                continue
            
            # Distribute tasks evenly across the time window
            interval = 1.0 / rate_limit.requests_per_second
            
            for i, task in enumerate(tasks):
                # Schedule tasks to maintain the desired rate
                task.scheduled_time = current_time + (i * interval)
                optimized_tasks.append(task)
        
        return optimized_tasks
    
    def shutdown(self):
        """Gracefully shutdown the scheduler."""
        self.running = False
        if self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        # Log final statistics
        logger.info("Task Scheduler Performance Summary:")
        for model, stats in self.get_model_performance().items():
            logger.info(f"  {model}: {stats['success_rate']:.1%} success rate, "
                       f"{stats['average_latency']:.2f}s average latency")

def create_optimized_experiment_runner(experiment: 'NewcombExperiment') -> SmartTaskScheduler:
    """Create an optimized task scheduler for the experiment."""
    # Get rate limits for all models
    rate_limits = {}
    for model in experiment.models:
        rate_limits[model] = get_rate_limit_for_model(model)
    
    # Create the scheduler
    scheduler = SmartTaskScheduler(rate_limits)
    
    # Return a wrapper function that integrates with the experiment
    def run_with_smart_scheduling(
        question_types=['cdt_capability', 'edt_capability', 'normative_attitude', 'personal_attitude'],
        repeats_per_model: int = 1,
        display_examples: bool = True
    ):
        # Create all tasks
        tasks = experiment._prepare_all_tasks(question_types, repeats_per_model)
        
        # Convert to ScheduledTask objects
        scheduled_tasks = []
        for i, task_info in enumerate(tasks):
            scheduled_task = ScheduledTask(
                task_id=f"task_{i}",
                model=task_info['model'],
                func=experiment._execute_single_task,
                args=(task_info,),
                kwargs={},
                priority=0  # Could be customized based on task type
            )
            scheduled_tasks.append(scheduled_task)
        
        # Optimize task distribution
        optimized_tasks = scheduler.optimize_task_distribution(scheduled_tasks)
        
        # Schedule all tasks
        scheduler.schedule_batch(optimized_tasks)
        
        # Wait for completion (simplified - you might want a more sophisticated approach)
        while True:
            with scheduler.queue_lock:
                if not scheduler.task_queue:
                    break
            time.sleep(1)
        
        # Collect results (this is simplified - adapt to your needs)
        results = {}
        for model in experiment.models:
            results[model] = []  # You'd populate this with actual results
        
        # Shutdown scheduler
        scheduler.shutdown()
        
        return results
    
    return run_with_smart_scheduling