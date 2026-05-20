# src/retry_handler.py

import random
import time
import logging
from typing import Callable, Any, Optional, List, Type
from functools import wraps

logger = logging.getLogger(__name__)

class RetryConfig:
    def __init__(
        self,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[List[Type[Exception]]] = None
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or [
            ConnectionError,
            TimeoutError,
            Exception  # Catch-all for HTTP 429, 500, etc.
        ]

def retry_with_exponential_backoff(config: RetryConfig = None):
    """Decorator for retrying functions with exponential backoff."""
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Check if this exception is retryable
                    if not any(isinstance(e, exc_type) for exc_type in config.retryable_exceptions):
                        logger.warning(f"Non-retryable exception: {type(e).__name__}: {str(e)}")
                        raise
                    
                    # Check if we've exhausted retries
                    if attempt >= config.max_retries:
                        logger.error(f"Max retries exceeded for {func.__name__}: {str(e)}")
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        config.base_delay * (config.exponential_base ** attempt),
                        config.max_delay
                    )
                    
                    # Add jitter if enabled
                    if config.jitter:
                        delay *= (0.5 + random.random())
                    
                    logger.info(f"Retry {attempt + 1}/{config.max_retries} for {func.__name__} after {delay:.2f}s: {str(e)}")
                    time.sleep(delay)
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator

def parse_retry_after_header(response_headers: dict) -> Optional[float]:
    """Parse the Retry-After header from HTTP response headers."""
    retry_after = response_headers.get('Retry-After') or response_headers.get('retry-after')
    
    if retry_after is None:
        return None
    
    try:
        # Try to parse as seconds
        return float(retry_after)
    except ValueError:
        # Try to parse as HTTP-date (not implemented here for brevity)
        logger.warning(f"Could not parse Retry-After header: {retry_after}")
        return None

class ProviderSpecificRetry:
    """Handle provider-specific retry logic based on error messages and headers."""
    
    @staticmethod
    def get_retry_delay(exception: Exception, response_headers: dict = None, provider: str = "") -> Optional[float]:
        """Get the retry delay based on the exception and provider-specific logic."""
        
        # Check for Retry-After header
        if response_headers:
            retry_after = parse_retry_after_header(response_headers)
            if retry_after:
                return retry_after
        
        # Provider-specific logic
        error_message = str(exception).lower()
        
        if provider.startswith('openai'):
            if 'rate limit' in error_message or '429' in error_message:
                return 60  # OpenAI suggests waiting 60s for rate limits
        
        elif provider.startswith('anthropic'):
            if 'rate limit' in error_message:
                return 30  # Conservative wait for Anthropic
        
        elif provider.startswith('google'):
            if 'quota exceeded' in error_message or 'resource exhausted' in error_message:
                return 60
        
        elif provider.startswith('xai'):
            if 'rate limit' in error_message:
                return 120  # xAI can have stricter limits
        
        elif provider.startswith('deepseek'):
            # DeepSeek queues requests, so shorter wait
            if 'rate limit' in error_message:
                return 30
        
        elif provider.startswith('alibaba'):
            if 'qps exceeded' in error_message or 'rate quota' in error_message:
                return 60
        
        return None  # Use default exponential backoff

def create_provider_retry_handler(provider: str) -> Callable:
    """Create a retry handler specific to a provider."""
    
    def retry_handler(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            max_retries = 5
            attempt = 0
            
            while attempt <= max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Extract response headers if available (depends on your client implementation)
                    response_headers = {}
                    if hasattr(e, 'response') and hasattr(e.response, 'headers'):
                        response_headers = dict(e.response.headers)
                    
                    # Get provider-specific retry delay
                    retry_delay = ProviderSpecificRetry.get_retry_delay(e, response_headers, provider)
                    
                    if attempt >= max_retries:
                        logger.error(f"Max retries exceeded for {provider}: {str(e)}")
                        raise
                    
                    if retry_delay:
                        logger.info(f"Provider-specific retry for {provider} after {retry_delay}s: {str(e)}")
                        time.sleep(retry_delay)
                    else:
                        # Fall back to exponential backoff
                        delay = min(2 ** attempt + random.uniform(0, 1), 60)
                        logger.info(f"Exponential backoff retry for {provider} after {delay:.2f}s: {str(e)}")
                        time.sleep(delay)
                    
                    attempt += 1
            
        return wrapper
    return retry_handler