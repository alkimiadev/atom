import random
import asyncio
import openai
from apikey import url, api_key
import logging # Add logging import
logger = logging.getLogger(__name__) # Add logger for this module

model_name = None

if isinstance(api_key, list):
	clients = [openai.AsyncClient(base_url=url, api_key=key) for key in api_key]
else:
	clients = [openai.AsyncClient(base_url=url, api_key=api_key)]

MAX_RETRIES = 3
total_prompt_tokens, total_completion_tokens, call_count, cost = 0, 0, 0, 0
current_prompt_tokens, current_completion_tokens = 0, 0

def set_model(model):
	global model_name
	model_name = model
	logger.info(f"LLM model set to: {model_name}") # Log model change

async def gen(msg, model=None, temperature=None, response_format="json_object"):
	global call_count, cost, current_prompt_tokens, current_completion_tokens, model_name
	if not model:
		model = model_name
	client = random.choice(clients)
	errors = []
	call_count += 1
	logger.debug(f"Initiating LLM call: model={model}, temp={temperature}, format={response_format}") # Log initiation

	DEFAULT_RETRY_AFTER = random.uniform(0.1, 2)
	for retry in range(MAX_RETRIES):
		try:
			async with asyncio.timeout(120 * 2):
				logger.debug(f"Attempting LLM call (Attempt {retry + 1}/{MAX_RETRIES})") # Log attempt start
				# Log prompt (potentially truncated)
				log_msg = msg[:500] + '...' if len(msg) > 500 else msg
				logger.debug(f"Prompt (truncated): {log_msg}") # Log prompt

				if model == "o3-mini":
					response = await client.chat.completions.create(
						model=model,
						messages=[
							{"role": "user", "content": msg},
						],
						# temperature=temperature,
						stop=None,
						# max_tokens=8192,
						response_format={"type": response_format}
					)
				else:
					response = await client.chat.completions.create(
						model=model,
						messages=[
							{"role": "user", "content": msg},
						],
						temperature=temperature,
						stop=None,
						#max_tokens=8192,
						response_format={"type": response_format}
					)
				content = response.choices[0].message.content
				
				usage = response.usage
				current_prompt_tokens = usage.prompt_tokens
				current_completion_tokens = usage.completion_tokens
				update_token()
				logger.info(f"LLM call successful (Attempt {retry + 1}). Tokens: Prompt={current_prompt_tokens}, Completion={current_completion_tokens}") # Log success and tokens
				# Log response (potentially truncated)
				log_content = content[:500] + '...' if len(content) > 500 else content
				logger.debug(f"Response (truncated): {log_content}") # Log response

				return content
		except asyncio.TimeoutError:
			error_msg = "Request timeout"
			errors.append(error_msg)
			logger.warning(f"LLM call failed (Attempt {retry + 1}): {error_msg}") # Log timeout
		except openai.RateLimitError:
			error_msg = "Rate limit error"
			errors.append(error_msg)
			logger.warning(f"LLM call failed (Attempt {retry + 1}): {error_msg}") # Log rate limit
		except openai.APIError as e:
			error_msg = f"API error: {str(e)}"
			errors.append(error_msg)
			logger.warning(f"LLM call failed (Attempt {retry + 1}): {error_msg}", exc_info=True) # Log API error
		except Exception as e:
			error_msg = f"Error: {type(e).__name__}, {str(e)}"
			errors.append(error_msg)
			logger.warning(f"LLM call failed (Attempt {retry + 1}): {error_msg}", exc_info=True) # Log other errors

		# Exponential backoff before retrying
		sleep_time = DEFAULT_RETRY_AFTER * (2 ** retry)
		logger.info(f"Retrying LLM call in {sleep_time:.2f} seconds...") # Log retry delay
		await asyncio.sleep(sleep_time)

	# This part is reached only if all retries fail
	logger.error(f"LLM call failed after {MAX_RETRIES} retries. Errors: {errors}") # Log final failure
	# Consider raising an exception here instead of returning None or an empty string
	# to signal the failure more explicitly to the caller.
	# For now, returning None as an indicator of complete failure.
	return None


def get_cost():
	return cost

def update_token():
	global total_prompt_tokens, total_completion_tokens, current_completion_tokens, current_prompt_tokens
	total_prompt_tokens += current_prompt_tokens
	total_completion_tokens += current_completion_tokens

def reset_token():
	global total_prompt_tokens, total_completion_tokens, call_count
	total_prompt_tokens = 0
	total_completion_tokens = 0
	call_count = 0

def get_token():
	return total_prompt_tokens, total_completion_tokens

def get_call_count():
	return call_count