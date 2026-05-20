# ./src/rate_limiter.py

import time
import asyncio
from typing import Dict, Optional, Any, Callable, Awaitable

class RateLimiter:
    """Rate limiter for API calls with per-provider configuration."""
    
    def __init__(self, provider_limits: Dict[str, Dict[str, float]]):
        """
        Initialize rate limiter with provider-specific limits.
        
        Args:
            provider_limits: Dictionary mapping provider names to their rate limits
                Example: {
                    "openai": {"requests_per_minute": 60, "tokens_per_minute": 10000},
                    "anthropic": {"requests_per_minute": 50, "tokens_per_minute": 8000},
                    ...
                }
        """
        self.provider_limits = provider_limits
        self.provider_locks = {provider: asyncio.Semaphore(max(1, int(limits["requests_per_minute"] / 60)))
                              for provider, limits in provider_limits.items()}
        self.last_request_time = {provider: {} for provider in provider_limits}
    
    def get_provider_from_model(self, model: str) -> str:
        """Extract provider from model string."""
        if ":" in model:
            return model.split(":")[0]
        elif model.startswith("gpt-") or model.startswith("chatgpt-") or model.startswith("o3-") or model.startswith("o4-"):
            return "openai"
        elif model.startswith("claude-"):
            return "anthropic"
        elif model.startswith("gemini-") or "gemini" in model:
            return "google"
        elif model.startswith("grok-") or "grok" in model:
            return "xai"
        elif "deepseek" in model:
            return "deepseek"
        elif "qwen" in model:
            return "alibaba"
        else:
            # Default to a conservative limit if provider unknown
            return "default"
    
    async def acquire(self, model: str) -> None:
        """Acquire permission to make an API call."""
        provider = self.get_provider_from_model(model)
        
        # Use default limits if provider not explicitly configured
        if provider not in self.provider_limits:
            provider = "default"
        
        # Get the lock for this provider
        lock = self.provider_locks.get(provider)
        if not lock:
            return  # No rate limiting for this provider
        
        # Wait for the semaphore (limits concurrent requests)
        await lock.acquire()
        
        # Calculate the minimum time between requests
        rpm = self.provider_limits[provider]["requests_per_minute"]
        min_delay = 60.0 / rpm if rpm > 0 else 0
        
        # Ensure we respect the minimum delay since last request
        if provider in self.last_request_time:
            model_key = model.replace(":", "_")
            last_time = self.last_request_time[provider].get(model_key, 0)
            elapsed = time.time() - last_time
            if elapsed < min_delay:
                await asyncio.sleep(min_delay - elapsed)
        
        # Update the last request time
        self.last_request_time[provider][model.replace(":", "_")] = time.time()
    
    def release(self, model: str) -> None:
        """Release the lock after API call is complete."""
        provider = self.get_provider_from_model(model)
        
        # Use default if provider not explicitly configured
        if provider not in self.provider_limits:
            provider = "default"
        
        # Release the lock
        lock = self.provider_locks.get(provider)
        if lock:
            lock.release()
    
    async def limited_call(self, model: str, func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        """Execute a function with rate limiting applied."""
        await self.acquire(model)
        try:
            return await func(*args, **kwargs)
        finally:
            self.release(model)

# Default rate limits based on known API provider constraints
DEFAULT_RATE_LIMITS = {
    "openai": {"requests_per_minute": 60, "tokens_per_minute": 10000},
    "anthropic": {"requests_per_minute": 50, "tokens_per_minute": 8000},
    "google": {"requests_per_minute": 120, "tokens_per_minute": 20000},
    "xai": {"requests_per_minute": 10, "tokens_per_minute": 5000},
    "deepseek": {"requests_per_minute": 60, "tokens_per_minute": 10000},
    "alibaba": {"requests_per_minute": 60, "tokens_per_minute": 10000},
    "default": {"requests_per_minute": 10, "tokens_per_minute": 5000}
}