import asyncio
import json
from functools import wraps
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple
import logging # Add logging import

# LLM interaction (LLMManager will be passed in)
# from llm import gen # Removed global import

# Prompters for different modules
from experiment.prompter import math, multichoice, multihop

# Utility functions
from experiment.utils import (
	extract_json,
	extract_xml,
	calculate_depth,
	score_math,
	score_mc,
	score_mh,
)

logger = logging.getLogger(__name__) # Add logger for this module

class AtomProcessor:
	'''
	Encapsulates the core logic for Atom of Thoughts processing (atom, plugin modes).
	Manages state like module type, prompter, scoring function, and retry counts.
	'''
	# Modified __init__ to accept llm_manager
	def __init__(self, llm_manager, max_retries: int = 5, label_retries: int = 3, atom_depth: int = 3):
		'''
		Initializes the AtomProcessor.

		Args:
			llm_manager: An instance of LLMManager from llm.py.
			max_retries: Default maximum retries for decorated functions.
			label_retries: Specific retry count for the labeling step in decompose.
			atom_depth: Default recursion depth for the atom method.
		'''
		if llm_manager is None:
			raise ValueError("llm_manager cannot be None")
		self.llm_manager = llm_manager # Store the LLMManager instance
		logger.info(f"Initializing AtomProcessor with LLMManager: max_retries={max_retries}, label_retries={label_retries}, atom_depth={atom_depth}")

		self.module_name: Optional[str] = None
		self.prompter: Optional[Any] = None  # Module for specific prompter functions
		self.score_func: Optional[callable] = None # Function for scoring results

		# Configuration for retries and depth
		self.max_retries: int = max_retries
		self.label_retries: int = label_retries
		self.atom_depth: int = atom_depth

		# State tracking
		self.failure_count: int = 0 # Tracks failures in retry decorator

	def configure_module(self, module_name: str):
		'''
		Configures the processor for a specific module type (e.g., 'math', 'multi-choice').
		Sets the appropriate prompter and scoring function.

		Args:
			module_name: The name of the module ('math', 'multi-choice', 'multi-hop').
		'''
		logger.info(f"Configuring module: {module_name}")
		self.module_name = module_name
		if module_name == 'math':
			self.prompter = math
			self.score_func = score_math
			logger.debug("Set prompter=math, score_func=score_math")
		elif module_name == 'multi-choice':
			self.prompter = multichoice
			self.score_func = score_mc
			logger.debug("Set prompter=multichoice, score_func=score_mc")
		elif module_name == 'multi-hop':
			self.prompter = multihop
			self.score_func = score_mh
			logger.debug("Set prompter=multihop, score_func=score_mh")
		else:
			# Consider raising an error for unknown module types
			logger.warning(f'Unknown module type configured: {module_name}. Prompter and score_func set to None.')
			self.prompter = None
			self.score_func = None

	# --- Decorator and Context Manager ---

	@staticmethod
	def retry(func_name):
		'''Decorator to retry LLM calls with checks.'''
		def decorator(func):
			@wraps(func)
			async def wrapper(*args, **kwargs):
				# Note: 'self' is the first arg because it's bound from the method call
				instance = args[0]
				actual_args = args[1:] # Exclude self for prompter call
				logger.debug(f"Entering retry wrapper for '{func_name}' with args: {actual_args}, kwargs: {kwargs}")

				retries = instance.max_retries
				last_result = {} # Keep track of the last result in case all retries fail

				while retries >= 0:
					attempt = instance.max_retries - retries + 1
					logger.debug(f"Retry wrapper for '{func_name}': Attempt {attempt}/{instance.max_retries + 1}")
					if not instance.prompter:
						logger.error("Prompter not configured in retry wrapper!")
						raise ValueError('Prompter not configured. Call configure_module first.')

					# Generate prompt using the instance's prompter
					prompt = getattr(instance.prompter, func_name)(*actual_args, **kwargs)

					# Always request text format, as we'll use extract_xml for parsing structured data
					response_format = 'text'
					# Call LLM using the instance's llm_manager
					response = await instance.llm_manager.gen(prompt, response_format=response_format)
					# --- Added detailed logging for contract raw response ---
					if func_name == 'contract':
						logger.info(f"Retry wrapper for 'contract': Raw LLM response received:\n--- START CONTRACT RESPONSE ---\n{response}\n--- END CONTRACT RESPONSE ---")
					else:
						logger.debug(f"Retry wrapper for '{func_name}': Raw response received (type: {type(response)}): {str(response)[:200]}...")
					# --- End added logging ---

					# Extract result based on format
					# Always use extract_xml now
					result = extract_xml(response)
					logger.debug(f"Retry wrapper for '{func_name}': Extracted XML result (type: {type(result)}): {result}")

					# Store raw response if result is a dict
					if isinstance(result, dict):
						result['response'] = response
						last_result = result # Update last successful-ish result
					else:
						last_result = {'response': response, 'error': 'Extraction failed'}
						# If extract_xml failed, result will be {} - handle this in check


					# Check if the result is valid using the prompter's check function
					is_valid = instance.prompter.check(func_name, result) # Check function needs to handle {} from failed extract_xml
					logger.debug(f"Retry wrapper for '{func_name}': Check result: {is_valid}")
					if is_valid:
						logger.debug(f"Retry wrapper for '{func_name}': Check passed. Returning result.")
						return result # Success

					# --- Check failed ---
					logger.warning(f"Retry wrapper for '{func_name}': Check failed for result (Attempt {attempt}). Retrying...")
					retries -= 1 # Decrement retries on failure

				# --- All retries failed ---
				logger.error(f"Retry wrapper for '{func_name}': All {instance.max_retries + 1} retry attempts failed.")
				if instance.max_retries > 1: # Only count failures if retries were enabled
					instance.failure_count += 1
				if instance.failure_count > 300: # Safety break
					logger.critical("Failure count exceeded safety break limit (300). Raising exception.")
					raise Exception('Too many failures across multiple calls')

				# Return the last result obtained, even if it failed the check
				logger.warning(f"Retry wrapper for '{func_name}': Returning last obtained result after all retries failed.")
				return last_result
			return wrapper
		return decorator

	@contextmanager
	def temporary_retries(self, value: int):
		'''Temporarily changes the max_retries setting.'''
		original = self.max_retries
		self.max_retries = value
		try:
			yield
		finally:
			self.max_retries = original

	# --- Core Logic Methods ---

	async def decompose(self, question: str, **kwargs) -> Dict[str, Any]:
		'''Decomposes a question into sub-questions with dependencies.'''
		logger.debug(f"Decompose started for question: {question[:100]}... , kwargs: {kwargs}")
		retries = self.label_retries
		if self.module_name == 'multi-hop':
			logger.debug("Decompose: Multi-hop path")
			if 'contexts' not in kwargs:
				logger.error("Multi-hop decompose called without 'contexts' in kwargs.")
				raise Exception('Multi-hop must have contexts')
			contexts = kwargs['contexts']
			# self.multistep might return a string or a dict
			logger.debug("Calling multistep for multi-hop...")
			multistep_output = await self.multistep(question, contexts)
			logger.debug(f"Multistep raw output (type {type(multistep_output)}): {str(multistep_output)[:200]}...")

			# --- Process Multistep Result (Handle String or Dict) ---
			multistep_result_dict = None
			# Keep original (as string) for logging if parsing fails. Handle potential non-string types.
			raw_multistep_for_log = str(multistep_output) if multistep_output is not None else "None"

			if isinstance(multistep_output, dict):
				# Use directly if it's already a dictionary
				logger.debug("Multistep output is already a dict.")
				multistep_result_dict = multistep_output
			elif isinstance(multistep_output, str):
				# Attempt to parse XML if it's a string
				logger.debug(f"Input to extract_xml (type {type(multistep_output)}): {str(multistep_output)[:500]}...") # ADDED
				# Attempt to parse XML if it's a string
				logger.debug("Attempting to parse multistep string output as XML.")
				multistep_result_dict = extract_xml(multistep_output) # Use extract_xml
				logger.debug(f"Output of extract_xml: {multistep_result_dict}") # ADDED
				try:
					# Check if extraction was successful (extract_xml returns {} on failure)
					if not multistep_result_dict: # Check for empty dict
						raise ValueError("extract_xml returned empty dict, indicating parsing failure.")
					logger.debug("Successfully parsed multistep XML string.")
				except Exception as e: # Catch potential errors during/after extraction
					# Log error but don't raise here. Let the check below handle the None result.
					logger.error(f"Failed to parse XML from multistep string output or validation failed: {e}", exc_info=False)
					logger.debug(f"Raw string causing multistep XML error: {raw_multistep_for_log}")
					# multistep_result_dict remains None
			else:
				# Handle unexpected types (including None if extract_json returned it)
				logger.warning(f"Unexpected type or None received from self.multistep: {type(multistep_output)}. Treating as failure.")
				# multistep_result_dict remains None

			# --- Check if a valid dictionary was obtained ---
			if multistep_result_dict is None:
				logger.error(f"Multistep processing failed to produce a valid dictionary. Raw output (truncated): {raw_multistep_for_log[:500]}")
				raise ValueError(f"Multistep processing failed to produce a valid dictionary. See logs. Raw output (truncated): {raw_multistep_for_log[:200]}")

			# --- Ensure the obtained dictionary has the required keys ---
			if 'sub-questions' not in multistep_result_dict:
				logger.error(f"Decomposition result is missing 'sub-questions' key. Result: {multistep_result_dict}")
				raise ValueError("Decomposition result missing 'sub-questions' key.")

			logger.debug(f"Processed multistep result (dict): {multistep_result_dict}")
			# Assign to the final variable name used later in the function
			decompose_result_dict = multistep_result_dict


			label_result = {}
			label_attempt = 0
			while retries > 0:
				label_attempt += 1
				logger.debug(f"Calling label for multi-hop (Attempt {label_attempt}/{self.label_retries}). Retries left: {retries}")
				# Pass the PARSED dictionary to label for multi-hop
				label_result = await self.label(question, multistep_result_dict)
				logger.debug(f"Label raw output (Attempt {label_attempt}): {label_result}")
				try:
					# Check if lengths match before accessing keys
					# Ensure label_result is also a dict (though label should return one)
					if not isinstance(label_result, dict) or 'sub-questions' not in label_result or 'sub-questions' not in multistep_result_dict:
						logger.warning(f"Label result missing 'sub-questions' or not a dict (Attempt {label_attempt}).")
						raise ValueError("Missing 'sub-questions' key in results or label_result is not a dict")

					label_sq_len = len(label_result.get('sub-questions', []))
					multistep_sq_len = len(multistep_result_dict.get('sub-questions', []))
					if label_sq_len != multistep_sq_len:
						logger.warning(f"Label sub-question length mismatch (Attempt {label_attempt}): Label={label_sq_len}, Multistep={multistep_sq_len}. Retrying label.")
						retries -= 1
						continue

					logger.debug(f"Label sub-question lengths match (Attempt {label_attempt}). Calculating depth...")
					calculate_depth(label_result['sub-questions']) # Check dependencies are valid
					logger.debug(f"Label check and depth calculation successful (Attempt {label_attempt}).")
					break # Success
				except Exception as e:
					logger.error(f"Labeling/Depth calculation failed (Attempt {label_attempt}): {e}", exc_info=True)
					retries -= 1
					if retries == 0:
						logger.error("Max label retries reached for multi-hop. Returning potentially incomplete/empty result.")
						# Return multistep result if labeling fails completely? Or raise?
						# For now, return the potentially incomplete label_result or empty if error
						return label_result if isinstance(label_result, dict) else {}
					logger.info(f"Waiting 1 second before retrying label...")
					await asyncio.sleep(1) # Small delay before retrying label
					continue

			# Combine results if successful
			logger.debug("Combining multistep and label results for multi-hop.")
			if 'sub-questions' in label_result and 'sub-questions' in multistep_result_dict:
				for i, (step, note) in enumerate(zip(multistep_result_dict['sub-questions'], label_result['sub-questions'])):
					# --- DEBUG LOGGING ---
					logger.debug(f"Processing zipped item #{i}: Type(note)={type(note)}, Value(note)='{str(note)[:100]}...'")
					# --- END DEBUG LOGGING ---
					step['depend'] = note.get('depend', []) # Use .get for safety
			logger.debug(f"Final multi-hop decompose result: {multistep_result_dict}")
			return multistep_result_dict

		else: # Math or Multi-choice
			logger.debug(f"Decompose: {self.module_name} path")
			logger.debug("Calling multistep...")
			multistep_result = await self.multistep(question)
			logger.debug(f"Multistep raw output: {multistep_result}")
			result = {}
			label_attempt = 0
			while retries > 0:
				label_attempt += 1
				logger.debug(f"Calling label for {self.module_name} (Attempt {label_attempt}/{self.label_retries}). Retries left: {retries}")
				# Pass response and answer strings to label
				result = await self.label(question, multistep_result.get('response', ''), multistep_result.get('answer', ''))
				logger.debug(f"Label raw output (Attempt {label_attempt}): {result}")
				try:
					if not isinstance(result, dict) or 'sub-questions' not in result:
						logger.warning(f"Label result missing 'sub-questions' or not a dict (Attempt {label_attempt}).")
						raise ValueError("Missing 'sub-questions' key in label result")
					logger.debug(f"Label result has 'sub-questions' (Attempt {label_attempt}). Calculating depth...")
					calculate_depth(result['sub-questions'])
					result['response'] = multistep_result.get('response', '') # Add original response back
					logger.debug(f"Label check and depth calculation successful (Attempt {label_attempt}).")
					break # Success
				except Exception as e:
					logger.error(f"Labeling/Depth calculation failed (Attempt {label_attempt}): {e}", exc_info=True)
					retries -= 1
					if retries == 0:
						logger.error(f"Max label retries reached for {self.module_name}. Returning last attempt.")
						return result if isinstance(result, dict) else {} # Return last attempt
					logger.info(f"Waiting 1 second before retrying label...")
					await asyncio.sleep(1)
					continue
			logger.debug(f"Final {self.module_name} decompose result: {result}")
			return result

	async def merging(self, question: str, decompose_result: dict, independent_subqs: list, dependent_subqs: list, **kwargs) -> Tuple[str, str, Dict[str, Any]]:
		'''Merges independent and dependent sub-questions via contraction.'''
		logger.debug(f"Merging started. Question: {question[:100]}..., Indep#: {len(independent_subqs)}, Dep#: {len(dependent_subqs)}, kwargs: {kwargs}")
		contract_args = [question, decompose_result, independent_subqs, dependent_subqs]
		if self.module_name == 'multi-hop':
			contract_args.append(kwargs.get('contexts')) # Use .get for safety
		logger.debug(f"Calling contract with args (excluding decompose_result): {[a for a in contract_args if not isinstance(a, dict)]}")

		contractd_result = await self.contract(*contract_args)
		logger.debug(f"Contract result: {contractd_result}")

		# Extract thought process and optimized question
		contractd_thought = contractd_result.get('response', '')
		contractd_question = contractd_result.get('question', '')
		logger.debug(f"Contracted thought (len {len(contractd_thought)}): {contractd_thought[:100]}...")
		logger.debug(f"Contracted question: {contractd_question}")

		# Solve the optimized question
		direct_args = [contractd_question]
		if self.module_name == 'multi-hop':
			# Pass context from contract result or original contexts
			direct_args.append(contractd_result.get('context', kwargs.get('contexts')))
		logger.debug(f"Calling direct with args: {direct_args}")

		contraction_result = await self.direct(*direct_args)
		logger.debug(f"Direct (contraction) result: {contraction_result}")

		logger.debug("Merging finished.")
		return contractd_thought, contractd_question, contraction_result

	async def atom(self, question: str, contexts: Optional[str] = None, direct_result: Optional[Dict] = None, decompose_result: Optional[Dict] = None, depth: Optional[int] = None, log: Optional[Dict] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
		'''
		Performs the Atom of Thoughts process by delegating to AtomStepExecutor.
		This method now acts as the entry point and orchestrator for the execution step.
		'''
		# Initialize the top-level log dictionary if not provided
		log = log if log is not None else {}
		# Determine the current execution level based on the log length
		level = len(log)

		logger.debug(f"AtomProcessor.atom called (level {level}). Instantiating AtomStepExecutor.")

		# Instantiate the executor for this step
		executor = AtomStepExecutor(
			processor=self,
			question=question,
			contexts=contexts,
			direct_result_in=direct_result, # Pass potentially pre-computed results
			decompose_result_in=decompose_result,
			depth_in=depth, # Pass the depth constraint for this level
			log_in=log,     # Pass the existing log
			level=level     # Pass the current level index
		)

		# Run the execution step
		final_result, updated_log = await executor.run()

		# --- Potential Future Recursion Logic ---
		# The original logic didn't implement recursion based on the result.
		# If recursion were desired, logic would go here to check `final_result['method']`,
		# potentially adjust inputs (e.g., use `final_result['answer']` as the new question),
		# decrement depth, and call `self.atom` again.
		# For now, we just return the result of the single execution step.
		# Example (Conceptual - DO NOT IMPLEMENT YET):
		# if final_result['method'] == 'decompose' and depth > 1:
		#     logger.info(f"AtomProcessor.atom (level {level}): Deeper recursion potentially needed...")
		#     # return await self.atom(question=..., contexts=..., ..., depth=depth-1, log=updated_log)

		logger.info(f"AtomProcessor.atom finished (level {level}). Returning method: {final_result.get('method')}")
		return final_result, updated_log


	async def plugin(self, question: str, contexts: Optional[str] = None, sample_num: int = 3) -> str:
		'''Generates multiple decompositions and selects the best contracted question.'''
		logger.info(f"Plugin started. Question: {question[:100]}..., Samples: {sample_num}")

		async def process_sample(sample_idx: int):
			'''Helper function to process one sample decomposition and contraction.'''
			logger.debug(f"Plugin: Starting process_sample {sample_idx+1}/{sample_num}")
			try:
				# Get decompose result
				decompose_args = {'contexts': contexts} if self.module_name == 'multi-hop' else {}
				logger.debug(f"Plugin sample {sample_idx+1}: Calling decompose...")
				decompose_result = await self.decompose(question, **decompose_args)
				logger.debug(f"Plugin sample {sample_idx+1}: Decompose result: {decompose_result}")

				if not decompose_result or 'sub-questions' not in decompose_result:
					logger.warning(f"Plugin sample {sample_idx+1}: Decomposition failed or missing sub-questions.")
					return None # Indicate failure for this sample

				# Separate independent and dependent sub-questions
				independent_subqs = [sub_q for sub_q in decompose_result['sub-questions'] if not sub_q.get('depend')]
				dependent_subqs = [sub_q for sub_q in decompose_result['sub-questions'] if sub_q.get('depend')]
				logger.debug(f"Plugin sample {sample_idx+1}: Separated sub-questions. Indep#: {len(independent_subqs)}, Dep#: {len(dependent_subqs)}")

				# Get contraction result
				merging_args = {
					'question': question,
					'decompose_result': decompose_result,
					'independent_subqs': independent_subqs,
					'dependent_subqs': dependent_subqs
				}
				if self.module_name == 'multi-hop':
					merging_args['contexts'] = contexts
				logger.debug(f"Plugin sample {sample_idx+1}: Calling merging...")
				contractd_thought, contractd_question, contraction_result = await self.merging(**merging_args)
				logger.debug(f"Plugin sample {sample_idx+1}: Merging successful. Contraction result: {contraction_result}")

				# Estimate token count (simple split for now)
				token_count = len(contraction_result.get('response', '').split())
				logger.debug(f"Plugin sample {sample_idx+1}: Estimated token count: {token_count}")

				return {
					# 'decompose_result': decompose_result, # Not needed for final selection
					'contractd_thought': contractd_thought,
					'contractd_question': contractd_question,
					'contraction_result': contraction_result,
					'token_count': token_count
				}
			except Exception as e:
				logger.error(f"Plugin sample {sample_idx+1}: Error processing sample: {e}", exc_info=True)
				return None # Indicate failure

		# Execute all samples in parallel
		logger.debug(f"Plugin: Creating {sample_num} process_sample tasks.")
		tasks = [process_sample(i) for i in range(sample_num)]
		logger.info(f"Plugin: Gathering results for {sample_num} samples...")
		all_results = await asyncio.gather(*tasks)
		valid_results = [r for r in all_results if r is not None] # Filter out failed samples
		logger.info(f"Plugin: Gathered {len(valid_results)} valid results out of {sample_num} samples.")

		if not valid_results:
			logger.error("Plugin: All samples failed processing. Falling back to original question.")
			# Fallback: return original question? Or raise error?
			return question # Return original question as fallback

		# Get direct result for original question (used for ensemble scoring)
		logger.debug("Plugin: Getting direct result for original question...")
		direct_args = [question]
		if self.module_name == 'multi-hop':
			direct_args.append(contexts)
		direct_result = await self.direct(*direct_args)
		logger.debug(f"Plugin: Direct result: {direct_result}")

		# Get ensemble result from all valid contracted results + direct result
		logger.debug("Plugin: Preparing ensemble...")
		all_responses = [direct_result.get('response', '')] + \
						[r['contraction_result'].get('response', '') for r in valid_results]
		ensemble_args = [question, [resp for resp in all_responses if resp]] # Filter empty responses
		logger.debug(f"Plugin: Ensemble input responses count: {len(ensemble_args[1])}")
		if self.module_name == 'multi-hop':
			ensemble_args.append(contexts)

		try:
			logger.debug("Plugin: Calling ensemble...")
			ensemble_result = await self.ensemble(*ensemble_args)
			ensemble_answer = ensemble_result.get('answer', '')
			logger.debug(f"Plugin: Ensemble successful. Answer: '{ensemble_answer}'")
		except Exception as e:
			logger.error(f"Plugin: Ensemble failed: {e}. Using direct answer as fallback.", exc_info=True)
			ensemble_answer = direct_result.get('answer', '') # Fallback scoring reference
			logger.debug(f"Plugin: Using fallback ensemble answer: '{ensemble_answer}'")


		# Calculate scores for each valid contracted result vs ensemble
		logger.debug("Plugin: Scoring valid contracted results against ensemble answer...")
		scores = []
		for i, result in enumerate(valid_results):
			contraction_answer = result['contraction_result'].get('answer')
			if contraction_answer is not None and self.score_func:
				try:
					score = self.score_func(contraction_answer, ensemble_answer)
					scores.append(score)
					logger.debug(f"Plugin: Score for sample {i+1} ('{contraction_answer}') vs ensemble ('{ensemble_answer}') = {score}")
				except Exception as e:
					logger.error(f"Plugin: Scoring failed for sample {i+1} result: {e}", exc_info=True)
					scores.append(0.0)
			else:
				logger.debug(f"Plugin: Assigning 0 score to sample {i+1} (answer missing or no score func).")
				scores.append(0.0) # Score 0 if answer missing or no score func
		logger.debug(f"Plugin: Scores: {scores}")

		# Find the best result(s) - those with the highest score
		if not scores:
			logger.error("Plugin: No valid scores calculated. Falling back.")
			# Fallback: return first valid result's question or original?
			fallback_question = valid_results[0]['contractd_question'] if valid_results else question
			logger.warning(f"Plugin: Returning fallback question: {fallback_question}")
			return fallback_question

		max_score = max(scores)
		best_indices = [i for i, s in enumerate(scores) if s == max_score]
		logger.debug(f"Plugin: Max score: {max_score}, Best indices: {best_indices}")

		# Among the best results, find the one with the lowest token count
		best_index_in_valid = min(best_indices, key=lambda i: valid_results[i]['token_count'])
		logger.debug(f"Plugin: Selected best index (lowest token count among max score): {best_index_in_valid}")

		# Return the best contracted question
		best_result = valid_results[best_index_in_valid]
		logger.info(f"Plugin finished. Selected contracted question from sample {best_index_in_valid + 1} with score {scores[best_index_in_valid]} and token count {best_result['token_count']}: {best_result['contractd_question']}")
		return best_result['contractd_question']


	# --- Methods decorated with retry ---
	# These methods primarily rely on the prompter call within the retry decorator

	@retry('direct')
	async def direct(self, question: str, contexts: Optional[str] = None):
		# The actual logic (prompt generation, LLM call, check) is handled by the retry decorator.
		# This method definition is mainly here to be decorated.
		# We can add pre/post processing if needed.
		if isinstance(question, (list, tuple)):
			# Handle cases where question might be passed as list (e.g., from MMLU config)
			question = '\n'.join(map(str, question)) # Simple join, adjust if needed
		# The decorator will use self.prompter.direct(question, contexts)
		pass # Decorator handles the core async call

	@retry('multistep')
	async def multistep(self, question: str, contexts: Optional[str] = None):
		# Logic handled by the retry decorator using self.prompter.multistep
		pass

	@retry('label')
	async def label(self, question: str, sub_questions_input: Any, answer: Optional[str] = None):
		# Logic handled by the retry decorator using self.prompter.label
		# The type of sub_questions_input depends on the module (dict for multi-hop, str otherwise)
		pass

	@retry('contract')
	async def contract(self, question: str, sub_result: dict, independent_subqs: list, dependent_subqs: list, contexts: Optional[str] = None):
		# Logic handled by the retry decorator using self.prompter.contract
		pass

	@retry('ensemble')
	async def ensemble(self, question: str, results: list, contexts: Optional[str] = None):
		# Logic handled by the retry decorator using self.prompter.ensemble
		pass

# ==============================================================================
# AtomStepExecutor Class - Encapsulates one level of Atom execution
# ==============================================================================

class AtomStepExecutor:
	'''
	Executes a single step/level of the Atom of Thoughts process.
	Encapsulates the logic previously found in AtomProcessor.atom.
	'''
	def __init__(self, processor: 'AtomProcessor', question: str, contexts: Optional[str] = None,
				 direct_result_in: Optional[Dict] = None, decompose_result_in: Optional[Dict] = None,
				 depth_in: Optional[int] = None, log_in: Optional[Dict] = None, level: int = 0):
		'''
		Initializes the AtomStepExecutor.

		Args:
			processor: Reference to the parent AtomProcessor instance.
			question: The question for this step.
			contexts: Optional context for multi-hop.
			direct_result_in: Optional pre-computed direct result.
			decompose_result_in: Optional pre-computed decompose result.
			depth_in: The maximum depth allowed for this execution path.
			log_in: The log dictionary from the parent level.
			level: The current execution level/index for logging.
		'''
		self.processor_ref: 'AtomProcessor' = processor
		self.question: str = question
		self.contexts: Optional[str] = contexts
		self.direct_result_in: Optional[Dict] = direct_result_in
		self.decompose_result_in: Optional[Dict] = decompose_result_in
		self.depth_in: Optional[int] = depth_in
		self.log_in: Optional[Dict] = log_in
		self.level: int = level

		# State for this step's execution
		self.log: Dict = log_in if log_in is not None else {}
		if self.level not in self.log:
			self.log[self.level] = {}

		self.direct_result: Optional[Dict] = None
		self.decompose_result: Optional[Dict] = None
		self.contraction_result: Optional[Dict] = None
		self.ensemble_result: Optional[Dict] = None
		self.scores: List[float] = []
		self.best_method: Optional[str] = None
		self.final_result_for_level: Optional[Dict] = None
		self.error: Optional[str] = None
		self.fallback_triggered: bool = False

		# Use a child logger for potentially clearer source identification
		self.logger = logger.getChild(self.__class__.__name__)
		self.logger.debug(f"Initialized (level {self.level}). Question: {self.question[:100]}..., Depth: {self.depth_in}")

	async def run(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
		'''
		Executes the full sequence of operations for one Atom step.

		Returns:
			A tuple containing the final result dictionary for this level
			and the updated log dictionary.
		'''
		self.logger.debug(f"Run started (level {self.level}).")

		if self._initialize_log_and_check_depth():
			self._update_log() # Update log with base case info
			self.logger.debug(f"Run finished early (level {self.level}): Depth limit reached.")
			return self.final_result_for_level, self.log

		if not await self._get_direct_and_decompose():
			self._update_log() # Update log with fallback info
			self.logger.debug(f"Run finished early (level {self.level}): Fallback during direct/decompose.")
			return self.final_result_for_level, self.log

		if not self._calculate_and_set_depth():
			self._update_log() # Update log with fallback info
			self.logger.debug(f"Run finished early (level {self.level}): Fallback during depth calculation.")
			return self.final_result_for_level, self.log

		if not await self._merge_subquestions():
			self._update_log() # Update log with fallback info
			self.logger.debug(f"Run finished early (level {self.level}): Fallback during merging.")
			return self.final_result_for_level, self.log

		if not await self._perform_ensemble():
			# Even if ensemble fails, we proceed to scoring using the fallback answer
			self.logger.warning(f"Run continuing (level {self.level}) despite ensemble fallback.")
			# Fallback is handled within _perform_ensemble, log updated there

		self._score_results()
		self._select_best_method()
		self._update_log()

		self.logger.debug(f"Run finished (level {self.level}). Returning method: {self.best_method}")
		return self.final_result_for_level, self.log

	# --- Private Helper Methods ---

	def _initialize_log_and_check_depth(self) -> bool:
		'''Handles log initialization and the depth base case check.'''
		# Log initialization is now done in __init__
		# Base case: depth limit reached
		if self.depth_in == 0:
			self.logger.info(f"(Level {self.level}): Depth limit reached. Returning base case.")
			self.best_method = 'depth_limit'
			self.final_result_for_level = {'method': 'depth_limit', 'response': None, 'answer': None}
			return True # Indicate depth limit reached
		return False # Depth limit not reached

	async def _get_direct_and_decompose(self) -> bool:
		'''Executes steps to get direct_result and decompose_result.'''
		self.logger.debug(f"(Level {self.level}): Step 1 - Getting Direct and Decompose results.")
		direct_args = [self.question]
		if self.processor_ref.module_name == 'multi-hop':
			direct_args.append(self.contexts)

		# Use provided results if available (for recursion), otherwise compute
		self.logger.debug(f"(Level {self.level}): Calling direct...")
		try:
			self.direct_result = self.direct_result_in if self.direct_result_in else await self.processor_ref.direct(*direct_args)
			self.logger.debug(f"(Level {self.level}): Direct result: {self.direct_result}")
		except Exception as e:
			# Handle potential error during direct call itself
			self._handle_error_and_fallback('direct_call', e)
			return False # Cannot proceed without direct result

		decompose_args = {'contexts': self.contexts} if self.processor_ref.module_name == 'multi-hop' else {}
		self.logger.debug(f"(Level {self.level}): Calling decompose...")
		try:
			decompose_output = self.decompose_result_in if self.decompose_result_in else await self.processor_ref.decompose(self.question, **decompose_args)
			self.logger.debug(f"(Level {self.level}): Decompose raw output (type {type(decompose_output)}): {str(decompose_output)[:200]}...")
		except Exception as e:
			# Handle potential error during decompose call itself
			self._handle_error_and_fallback('decompose_call', e)
			return False

		# --- Process Decompose Result (Handle String or Dict) ---
		raw_decompose_for_log = decompose_output # Keep original for logging if parsing fails
		if isinstance(decompose_output, str):
			try:
				self.logger.debug(f"(Level {self.level}): Attempting to parse decompose string output as JSON.")
				self.decompose_result = json.loads(decompose_output)
				self.logger.debug(f"(Level {self.level}): Successfully parsed decompose JSON string.")
			except json.JSONDecodeError as e:
				self._handle_error_and_fallback('decompose_parsing', e, raw_data=raw_decompose_for_log)
				return False # Indicate failure/fallback
		elif isinstance(decompose_output, dict):
			self.logger.debug(f"(Level {self.level}): Decompose output is already a dict.")
			self.decompose_result = decompose_output
		else:
			self._handle_error_and_fallback('decompose_unexpected_type', TypeError(f'Unexpected type from decompose: {type(decompose_output)}'), raw_data=raw_decompose_for_log)
			return False # Indicate failure/fallback

		# Handle potential failure in decompose or missing 'sub-questions' key after processing
		if not self.decompose_result or 'sub-questions' not in self.decompose_result:
			err_msg = 'Decomposition result missing sub-questions key or is invalid.'
			self.logger.error(f"(Level {self.level}): {err_msg} Result: {self.decompose_result}")
			self._handle_error_and_fallback('decompose_invalid_structure', ValueError(err_msg), raw_data=raw_decompose_for_log)
			return False # Indicate failure/fallback

		self.logger.debug(f"(Level {self.level}): Processed decompose result (dict): {self.decompose_result}")
		return True # Indicate success

	def _calculate_and_set_depth(self) -> bool:
		'''Calculates depth from decompose_result and determines the effective depth.'''
		self.logger.debug(f"(Level {self.level}): Step 2 - Setting recursion depth.")
		try:
			# Ensure sub-questions is a list before passing to calculate_depth
			sub_questions = self.decompose_result.get('sub-questions')
			if not isinstance(sub_questions, list):
				raise TypeError(f"'sub-questions' key does not contain a list. Found: {type(sub_questions)}")
			current_depth = calculate_depth(sub_questions)
			self.logger.debug(f"(Level {self.level}): Calculated depth from decompose result: {current_depth}")
		except Exception as e:
			self._handle_error_and_fallback('depth_calculation', e, raw_data=self.decompose_result)
			return False # Indicate failure/fallback

		# Determine depth for this level: minimum of default, calculated, or passed-in depth
		effective_depth = self.depth_in if self.depth_in is not None else self.processor_ref.atom_depth
		effective_depth = min(effective_depth, current_depth)
		# Note: We don't store effective_depth as an attribute as it's only used for potential recursion,
		# which is not fully implemented in the original logic we are refactoring.
		self.logger.debug(f"(Level {self.level}): Effective depth for this level set to: {effective_depth}")
		return True # Indicate success

	async def _merge_subquestions(self) -> bool:
		'''Separates sub-questions and performs Merging (Contraction).'''
		self.logger.debug(f"(Level {self.level}): Step 3 - Separating sub-questions and merging.")
		sub_questions_list = self.decompose_result.get('sub-questions', []) # Use .get for safety
		self.logger.debug(f"(Level {self.level}): Sub-questions list type: {type(sub_questions_list)}, Content (first 500 chars): {str(sub_questions_list)[:500]}")

		independent_subqs = []
		dependent_subqs = []
		if isinstance(sub_questions_list, list):
			for i, sub_q in enumerate(sub_questions_list):
				self.logger.debug(f"(Level {self.level}): Processing sub_q #{i}: Type={type(sub_q)}, Value={str(sub_q)[:200]}")
				try:
					if isinstance(sub_q, dict):
						# Check for the 'depend' key and if it's empty/falsey
						if not sub_q.get('depend'):
							independent_subqs.append(sub_q)
						else:
							dependent_subqs.append(sub_q)
					else:
						self.logger.warning(f"(Level {self.level}): Sub_q #{i} is not a dictionary, skipping dependency check.")
				except Exception as e:
					# Log error processing a specific sub-question but continue if possible
					self.logger.error(f"(Level {self.level}): Error processing sub_q #{i}: {e}", exc_info=True)
		else:
			# If 'sub-questions' is not a list, we cannot proceed with merging logic.
			self.logger.error(f"(Level {self.level}): 'sub-questions' is not a list, cannot separate for merging.")
			self._handle_error_and_fallback('merge_invalid_subquestions_list', TypeError("'sub-questions' is not a list"))
			return False # Indicate failure/fallback

		self.logger.debug(f"(Level {self.level}): Separated sub-questions. Independent: {len(independent_subqs)}, Dependent: {len(dependent_subqs)}")

		merging_args = {
			'question': self.question,
			'decompose_result': self.decompose_result, # Use the processed dictionary
			'independent_subqs': independent_subqs,
			'dependent_subqs': dependent_subqs
		}
		if self.processor_ref.module_name == 'multi-hop':
			merging_args['contexts'] = self.contexts

		try:
			self.logger.debug(f"(Level {self.level}): Calling merging...")
			contractd_thought, contractd_question, self.contraction_result = await self.processor_ref.merging(**merging_args)
			self.logger.debug(f"(Level {self.level}): Merging successful. Contraction result: {self.contraction_result}")

			# Augment contraction result for logging and potential use
			if isinstance(self.contraction_result, dict):
				self.contraction_result['contraction_thought'] = contractd_thought
				# Keep original augmentation for logging consistency
				self.contraction_result['sub-questions'] = independent_subqs + [{
					'description': contractd_question,
					'response': self.contraction_result.get('response', ''),
					'answer': self.contraction_result.get('answer', ''),
					'depend': [] # Implicit dependency
				}]
				self.logger.debug(f"(Level {self.level}): Augmented contraction result: {self.contraction_result}")
			else:
				# If merging didn't return a dict, treat it as an error.
				raise TypeError(f"Merging result is not a dictionary: {type(self.contraction_result)}")

		except Exception as e:
			self._handle_error_and_fallback('merging', e)
			return False # Indicate failure/fallback

		return True # Indicate success

	async def _perform_ensemble(self) -> bool:
		'''Executes the ensemble step.'''
		self.logger.debug(f"(Level {self.level}): Step 4 - Performing ensemble.")
		ensemble_args = [self.question]
		# Ensure responses exist and are strings before appending
		results_for_ensemble = [self.direct_result, self.decompose_result, self.contraction_result]
		responses_for_ensemble = []
		for res in results_for_ensemble:
			if res and isinstance(res, dict):
				response = res.get('response')
				if isinstance(response, str): # Ensure it's a string
					responses_for_ensemble.append(response)
				elif response is not None:
					self.logger.warning(f"(Level {self.level}): Non-string response found for ensemble: {type(response)}. Converting to string.")
					responses_for_ensemble.append(str(response))

		ensemble_args.append(responses_for_ensemble)
		self.logger.debug(f"(Level {self.level}): Ensemble input responses count: {len(responses_for_ensemble)}")

		if self.processor_ref.module_name == 'multi-hop':
			ensemble_args.append(self.contexts)

		try:
			self.logger.debug(f"(Level {self.level}): Calling ensemble...")
			self.ensemble_result = await self.processor_ref.ensemble(*ensemble_args)
			self.logger.debug(f"(Level {self.level}): Ensemble successful. Result: {self.ensemble_result}")
			# Ensure ensemble result is a dict
			if not isinstance(self.ensemble_result, dict):
				raise TypeError(f"Ensemble result is not a dictionary: {type(self.ensemble_result)}")

		except Exception as e:
			self._handle_error_and_fallback('ensemble', e)
			# Use direct answer as fallback for ensemble answer
			ensemble_answer_fallback = self.direct_result.get('answer') if self.direct_result and isinstance(self.direct_result, dict) else None
			self.ensemble_result = {'answer': ensemble_answer_fallback} # Set fallback result structure
			self.logger.warning(f"(Level {self.level}): Using fallback ensemble answer: '{ensemble_answer_fallback}'")
			# We don't return False here, as scoring can still proceed with the fallback answer.

		return True # Indicate success (even if fallback was used internally for the answer)

	def _score_results(self):
		'''Scores results against the ensemble answer.'''
		self.logger.debug(f"(Level {self.level}): Step 5 - Scoring results.")
		self.scores = []
		results_to_score = [self.direct_result, self.decompose_result, self.contraction_result]
		# Safely get ensemble answer
		ensemble_answer = None
		if self.ensemble_result and isinstance(self.ensemble_result, dict):
			ensemble_answer = self.ensemble_result.get('answer')

		score_func = self.processor_ref.score_func

		if ensemble_answer is not None and score_func:
			self.logger.debug(f"(Level {self.level}): Scoring against ensemble answer: '{ensemble_answer}'")
			# Check if all valid results have the same answer as the ensemble
			valid_answers = []
			for res in results_to_score:
				if res and isinstance(res, dict) and 'answer' in res:
					valid_answers.append(res.get('answer'))

			# Check if valid_answers is not empty before the 'all' check
			if valid_answers and all(ans == ensemble_answer for ans in valid_answers):
				self.scores = [1.0] * len(results_to_score) # Assign perfect score if all match ensemble
				self.logger.debug(f"(Level {self.level}): All valid answers matched ensemble. Scores: {self.scores}")
			else:
				self.logger.debug(f"(Level {self.level}): Scoring each result individually.")
				for i, result in enumerate(results_to_score):
					method_name = ['direct', 'decompose', 'contract'][i]
					result_answer = None
					if result and isinstance(result, dict) and 'answer' in result:
						result_answer = result.get('answer')

					if result_answer is not None:
						try:
							score_value = score_func(result_answer, ensemble_answer)
							self.scores.append(score_value)
							self.logger.debug(f"(Level {self.level}): Score for {method_name} ('{result_answer}') vs ensemble ('{ensemble_answer}') = {score_value}")
						except Exception as e:
							self.logger.error(f"(Level {self.level}): Scoring failed for {method_name} result: {e}", exc_info=True)
							self.scores.append(0.0) # Assign 0 score on error
					else:
						self.logger.debug(f"(Level {self.level}): Assigning 0 score to {method_name} (invalid result or missing/None answer).")
						self.scores.append(0.0) # Assign 0 score if result is invalid or missing answer
		else:
			if ensemble_answer is None:
				self.logger.warning(f"(Level {self.level}): Cannot score results because ensemble answer is None.")
			if not score_func:
				self.logger.warning(f"(Level {self.level}): Cannot score results because score_func is not set.")
			self.scores = [0.0] * len(results_to_score) # Assign 0 scores if scoring is not possible

		# Ensure scores list matches the number of results
		while len(self.scores) < len(results_to_score):
			self.scores.append(0.0)
		self.logger.debug(f"(Level {self.level}): Final scores: {self.scores}")

	def _select_best_method(self):
		'''Selects the best method based on scores or fallback conditions.'''
		self.logger.debug(f"(Level {self.level}): Step 7 - Selecting best method.")
		methods = {
			0: ('direct', self.direct_result),
			1: ('decompose', self.decompose_result),
			2: ('contract', self.contraction_result),
			# Ensemble result itself isn't usually the final choice unless others fail badly,
			# but we keep it as a potential fallback target if scoring leads nowhere.
			-1: ('ensemble_fallback', self.ensemble_result)
		}

		# If a fallback was triggered earlier, the final result is already set
		if self.fallback_triggered:
			self.best_method = 'direct_fallback'
			# final_result_for_level is already set in _handle_error_and_fallback
			self.logger.info(f"(Level {self.level}): Fallback previously triggered. Final method: {self.best_method}")
			return # Exit early as decision is made

		# If depth limit reached, decision is already made
		if self.depth_in == 0:
			self.best_method = 'depth_limit'
			# final_result_for_level is already set in _initialize_log_and_check_depth
			self.logger.info(f"(Level {self.level}): Depth limit reached. Final method: {self.best_method}")
			return # Exit early

		# Select based on scores
		best_method_idx = -1
		if self.scores: # Check if scores list is not empty and contains non-zero scores
			max_score = max(self.scores)
			# Only consider scores > 0? Or just the max? Let's stick to max for now.
			# Find the first index matching the max score (priority: direct > decompose > contract)
			try:
				best_method_idx = self.scores.index(max_score)
			except ValueError:
				# This should not happen if self.scores is not empty, but handle defensively.
				self.logger.warning(f"(Level {self.level}): Max score not found in scores list {self.scores}. Defaulting.")
				best_method_idx = -1 # Default to ensemble fallback below
		else:
			self.logger.warning(f"(Level {self.level}): Scores list is empty. Defaulting.")
			best_method_idx = -1 # Default to ensemble fallback

		# Get the selected method name and result dict
		selected_method_name, selected_result_dict = methods.get(best_method_idx, methods[-1])

		# Ensure the selected result is a dictionary, default to empty if not
		if not isinstance(selected_result_dict, dict):
			self.logger.warning(f"(Level {self.level}): Selected result for method '{selected_method_name}' is not a dictionary ({type(selected_result_dict)}). Using empty dict.")
			selected_result_dict = {}

		self.best_method = selected_method_name
		# Construct the final result structure expected by the caller
		self.final_result_for_level = {
			'method': self.best_method,
			'response': selected_result_dict.get('response'),
			'answer': selected_result_dict.get('answer'),
		}

		score_info = f"Score: {self.scores[best_method_idx]}" if best_method_idx != -1 and best_method_idx < len(self.scores) else 'N/A or Fallback'
		self.logger.info(f"(Level {self.level}): Selected best method: {self.best_method} (Index: {best_method_idx}, {score_info})")


	def _update_log(self):
		'''Populates the log dictionary for this level with final decisions and intermediate results.'''
		self.logger.debug(f"(Level {self.level}): Step 8 - Updating log dictionary.")
		# Ensure log entry for this level exists
		if self.level not in self.log:
			self.log[self.level] = {}

		log_entry = {
			'method': self.best_method, # Final selected method
			'scores': self.scores,
			'direct': self.direct_result,
			'decompose': self.decompose_result,
			'contract': self.contraction_result,
			'ensemble': self.ensemble_result, # Log ensemble result too
		}
		if self.error:
			log_entry['error'] = self.error
		if self.fallback_triggered:
			log_entry['fallback_triggered'] = True

		self.log[self.level].update(log_entry)
		self.logger.debug(f"(Level {self.level}): Log updated.")


	def _handle_error_and_fallback(self, step_name: str, error: Exception, raw_data: Any = None):
		'''Centralized function to log errors, trigger fallback, and set final result.'''
		error_msg = f'{step_name} failed: {error}'
		self.logger.error(f"(Level {self.level}): {error_msg}", exc_info=True) # Log with traceback
		self.error = error_msg # Store the error message for logging
		self.fallback_triggered = True

		# Ensure log entry for this level exists before updating
		if self.level not in self.log:
			self.log[self.level] = {}

		# Store relevant data in log for debugging
		log_update = {
			'error': self.error,
			'fallback_triggered': True,
			# Log direct result as it's the fallback target
			'direct': self.direct_result,
		}
		# Add specific context based on the step that failed
		if step_name in ['decompose_call', 'decompose_parsing', 'decompose_unexpected_type', 'decompose_invalid_structure']:
			log_update['raw_decompose'] = raw_data
			# Log parsed decompose only if it exists and wasn't the source of the error
			if step_name != 'decompose_parsing' and self.decompose_result is not None:
				log_update['parsed_decompose'] = self.decompose_result
		elif step_name == 'depth_calculation':
			log_update['decompose'] = self.decompose_result
		elif step_name in ['merging', 'merge_invalid_subquestions_list']:
			log_update['decompose'] = self.decompose_result
		elif step_name == 'ensemble':
			# Log inputs to ensemble if available
			log_update['decompose'] = self.decompose_result
			log_update['contract'] = self.contraction_result

		self.log[self.level].update(log_update)

		# Set final result to direct result as fallback
		self.logger.warning(f"(Level {self.level}): Falling back to direct result due to {step_name} error.")
		# Safely get response/answer from direct_result
		direct_response = None
		direct_answer = None
		if self.direct_result and isinstance(self.direct_result, dict):
			direct_response = self.direct_result.get('response')
			direct_answer = self.direct_result.get('answer')

		self.final_result_for_level = {
			'method': 'direct_fallback',
			'response': direct_response,
			'answer': direct_answer,
		}
		self.best_method = 'direct_fallback' # Ensure best_method reflects the fallback
