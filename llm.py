import random
import asyncio
import openai
from apikey import url, api_key  # Assuming apikey.py exists and is configured
import logging

logger = logging.getLogger(__name__)

# --- Module-Level Setup ---

# Configure OpenAI clients based on api_key
try:
    if isinstance(api_key, list):
        clients = [openai.AsyncClient(base_url=url, api_key=key) for key in api_key]
        logger.info(f'Initialized {len(clients)} OpenAI clients with multiple API keys.')
    elif isinstance(api_key, str):
        clients = [openai.AsyncClient(base_url=url, api_key=api_key)]
        logger.info('Initialized 1 OpenAI client with a single API key.')
    else:
        clients = []
        logger.error('Invalid api_key format in apikey.py. Expected string or list.')
        # Potentially raise an error here or handle downstream
except NameError:
    clients = []
    logger.error("Could not find 'api_key' or 'url' in apikey.py. Please ensure it exists and is configured.")
    # Potentially raise an error here
except Exception as e:
    clients = []
    logger.error(f'Error initializing OpenAI clients: {e}', exc_info=True)
    # Potentially raise an error here

MAX_RETRIES = 3
DEFAULT_TIMEOUT = 240 # Default timeout for API calls in seconds (120 * 2)

# --- LLMManager Class ---

class LLMManager:
    """
    Manages interactions with the LLM API, including client selection,
    retries, state (model name), and usage statistics tracking.
    """
    def __init__(self, default_model=None):
        """
        Initializes the LLMManager.

        Args:
            default_model (str, optional): The default model name to use if not
                                           specified in gen(). Defaults to None.
        """
        self.model_name = default_model
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.call_count = 0
        self.cost = 0  # Placeholder for cost tracking
        self.current_prompt_tokens = 0
        self.current_completion_tokens = 0
        if clients:
             logger.info(f"LLMManager initialized. Default model: {self.model_name or 'Not set'}")
        else:
             logger.warning('LLMManager initialized, but no OpenAI clients are available. LLM calls will fail.')


    def set_model(self, model):
        """Sets the default model name for subsequent calls to gen()."""
        if not isinstance(model, str) or not model:
            logger.error(f'Invalid model name provided: {model}. Model name must be a non-empty string.')
            return # Or raise ValueError
        self.model_name = model
        logger.info(f'Default LLM model set to: {self.model_name}')

    async def gen(self, msg, model=None, temperature=None, response_format='json_object', timeout=DEFAULT_TIMEOUT):
        """
        Generates text using the configured LLM.

        Args:
            msg (str): The input prompt message.
            model (str, optional): The specific model to use for this generation.
                                   Overrides the default model if set. Defaults to None.
            temperature (float, optional): Sampling temperature. Defaults to None.
                                          Ignored for 'o3-mini' model.
            response_format (str, optional): The desired response format (e.g., "text", "json_object").
                                             Defaults to "json_object".
            timeout (int, optional): Timeout in seconds for the API call attempt.
                                     Defaults to DEFAULT_TIMEOUT.

        Returns:
            str: The generated content, or None if generation failed after retries.

        Raises:
            ValueError: If no model is specified (neither default nor per-call)
                        or if no OpenAI clients are available.
        """
        if not clients:
            logger.error('Cannot generate text: No OpenAI clients available.')
            raise ValueError('No OpenAI clients available.')

        effective_model = model if model else self.model_name
        if not effective_model:
            logger.error('LLM model name not set. Call set_model() first or provide \'model\' in gen().')
            raise ValueError('LLM model name not set.')

        # Ensure response_format is valid
        valid_response_formats = ['text', 'json_object']
        if response_format not in valid_response_formats:
             logger.warning(f"Invalid response_format '{response_format}'. Defaulting to 'text'.")
             response_format = 'text'

        client = random.choice(clients)
        errors = []
        self.call_count += 1
        logger.debug(f'Initiating LLM call #{self.call_count}: model={effective_model}, temp={temperature}, format={response_format}')

        DEFAULT_RETRY_AFTER = random.uniform(0.1, 2) # Base delay for backoff

        for retry in range(MAX_RETRIES):
            try:
                async with asyncio.timeout(timeout):
                    logger.debug(f'Attempting LLM call (Attempt {retry + 1}/{MAX_RETRIES})')
                    # Log prompt safely (truncated)
                    log_msg = str(msg)[:500] + ('...' if len(str(msg)) > 500 else '')
                    logger.debug(f'Prompt (truncated): {log_msg}')

                    request_params = {
                        'model': effective_model,
                        'messages': [{'role': 'user', 'content': msg}],
                        'stop': None, # Add stop sequences if needed
                        'response_format': {'type': response_format}
                        # Consider adding max_tokens if necessary
                    }

                    # Apply temperature only if it's not None and model is not o3-mini
                    if temperature is not None:
                        if effective_model != 'o3-mini':
                            request_params['temperature'] = temperature
                        else:
                            logger.warning(f"Temperature parameter ignored for model 'o3-mini'.")

                    response = await client.chat.completions.create(**request_params)

                    content = response.choices[0].message.content
                    usage = response.usage

                    # Safely access usage data
                    self.current_prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                    self.current_completion_tokens = getattr(usage, 'completion_tokens', 0)
                    self.update_token() # Update totals

                    logger.info(f'LLM call successful (Attempt {retry + 1}). Tokens: Prompt={self.current_prompt_tokens}, Completion={self.current_completion_tokens}')
                    # Log response safely (truncated)
                    log_content = str(content)[:500] + ('...' if len(str(content)) > 500 else '')
                    logger.debug(f'Response (truncated): {log_content}')
                    return content # Success

            except asyncio.TimeoutError:
                error_msg = f'Request timed out after {timeout} seconds'
                errors.append(error_msg)
                logger.warning(f'LLM call failed (Attempt {retry + 1}/{MAX_RETRIES}): {error_msg}')
            except openai.RateLimitError as e:
                error_msg = f'Rate limit error: {e}'
                errors.append(error_msg)
                logger.warning(f'LLM call failed (Attempt {retry + 1}/{MAX_RETRIES}): {error_msg}')
                # Consider specific backoff for rate limits based on headers if available
            except openai.APIConnectionError as e:
                 error_msg = f'API connection error: {e}'
                 errors.append(error_msg)
                 logger.warning(f'LLM call failed (Attempt {retry + 1}/{MAX_RETRIES}): {error_msg}', exc_info=True)
            except openai.APIStatusError as e: # Catch broader API errors (e.g., 5xx)
                 error_msg = f'API status error ({e.status_code}): {e.message}'
                 errors.append(error_msg)
                 logger.warning(f'LLM call failed (Attempt {retry + 1}/{MAX_RETRIES}): {error_msg}', exc_info=True)
            except openai.APIError as e: # Catch-all for other OpenAI specific errors
                error_msg = f'Generic API error: {e}'
                errors.append(str(error_msg))
                logger.warning(f'LLM call failed (Attempt {retry + 1}/{MAX_RETRIES}): {error_msg}', exc_info=True)
            except Exception as e: # Catch unexpected errors
                error_msg = f'Unexpected error: {type(e).__name__}: {str(e)}'
                errors.append(error_msg)
                logger.error(f'LLM call failed (Attempt {retry + 1}/{MAX_RETRIES}) with unexpected error: {error_msg}', exc_info=True) # Log as error

            # Exponential backoff before retrying (if not the last retry)
            if retry < MAX_RETRIES - 1:
                sleep_time = DEFAULT_RETRY_AFTER * (2 ** retry)
                logger.info(f'Retrying LLM call in {sleep_time:.2f} seconds...')
                await asyncio.sleep(sleep_time)

        # This part is reached only if all retries fail
        logger.error(f'LLM call failed definitively after {MAX_RETRIES} retries for model {effective_model}. Errors: {errors}')
        # Consider raising a custom exception to signal failure clearly
        # class LLMGenerationError(Exception): pass
        # raise LLMGenerationError(f"Failed after {MAX_RETRIES} retries: {errors}")
        return None # Return None to indicate failure

    def get_cost(self):
        """Returns the placeholder cost value."""
        # Placeholder: Implement actual cost calculation based on tokens and model if needed
        logger.warning('Cost calculation is not implemented. Returning placeholder value.')
        return self.cost

    def update_token(self):
        """Updates total token counts with the counts from the last successful call."""
        self.total_prompt_tokens += self.current_prompt_tokens
        self.total_completion_tokens += self.current_completion_tokens
        # Reset current tokens after adding to total? Optional.
        # self.current_prompt_tokens = 0
        # self.current_completion_tokens = 0

    def reset_token(self):
        """Resets the total token counts and call count to zero."""
        logger.info(f'Resetting token counts. Previous totals: Prompt={self.total_prompt_tokens}, Completion={self.total_completion_tokens}, Calls={self.call_count}')
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.call_count = 0
        self.current_prompt_tokens = 0 # Also reset current tokens
        self.current_completion_tokens = 0

    def get_token(self):
        """Returns the total accumulated prompt and completion tokens."""
        return self.total_prompt_tokens, self.total_completion_tokens

    def get_call_count(self):
        """Returns the total number of calls made to gen()."""
        return self.call_count