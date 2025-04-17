import asyncio
import json
from functools import wraps
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple
import logging # Add logging import

# LLM interaction
from llm import gen

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
	def __init__(self, max_retries: int = 5, label_retries: int = 3, atom_depth: int = 3):
		'''
		Initializes the AtomProcessor.

		Args:
			max_retries: Default maximum retries for decorated functions.
			label_retries: Specific retry count for the labeling step in decompose.
			atom_depth: Default recursion depth for the atom method.
		'''
		logger.info(f"Initializing AtomProcessor: max_retries={max_retries}, label_retries={label_retries}, atom_depth={atom_depth}")
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

					# Determine response format based on module and function
					response_format = 'json_object'
					if instance.module_name != 'multi-hop' or func_name == 'contract':
						if func_name != 'label':
							response_format = 'text'

					# Call LLM
					response = await gen(prompt, response_format=response_format)
					logger.debug(f"Retry wrapper for '{func_name}': Raw response received (type: {type(response)}): {str(response)[:200]}...")

					# Extract result based on format
					if response_format == 'json_object':
						result = extract_json(response)
					else: # text format
						result = extract_xml(response)
					logger.debug(f"Retry wrapper for '{func_name}': Extracted result (type: {type(result)}): {result}")

					# Store raw response if result is a dict
					if isinstance(result, dict):
						result['response'] = response
						last_result = result # Update last successful-ish result
					else:
						last_result = {'response': response, 'error': 'Extraction failed'}


					# Check if the result is valid using the prompter's check function
					is_valid = instance.prompter.check(func_name, result)
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
				try:
					# Attempt to parse if it's a string
					logger.debug("Attempting to parse multistep string output as JSON.")
					multistep_result_dict = json.loads(multistep_output)
					logger.debug("Successfully parsed multistep JSON string.")
				except json.JSONDecodeError as e:
					# Log error but don't raise here. Let the check below handle the None result.
					logger.error(f"Failed to parse JSON from multistep string output: {e}", exc_info=False) # exc_info=False to avoid duplicate traceback
					logger.debug(f"Raw string causing multistep JSON error: {raw_multistep_for_log}")
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
		'''Performs the Atom of Thoughts process recursively.'''
		# Initialize logging for this level
		log = log if log is not None else {}
		index = len(log)
		log[index] = {}
		logger.debug(f"Atom started (level {index}). Question: {question[:100]}..., Depth: {depth}")

		# Base case: depth limit reached
		if depth == 0:
			logger.info(f"Atom (level {index}): Depth limit reached. Returning base case.")
			# Return format consistent with recursive step but indicate termination
			return {'method': 'depth_limit', 'response': None, 'answer': None}, log

		# --- Step 1: Get results from Direct and Decompose approaches ---
		logger.debug(f"Atom (level {index}): Step 1 - Getting Direct and Decompose results.")
		direct_args = [question]
		if self.module_name == 'multi-hop':
			direct_args.append(contexts)
		# Use provided results if available (for recursion), otherwise compute
		logger.debug(f"Atom (level {index}): Calling direct...")
		direct_result = direct_result if direct_result else await self.direct(*direct_args)
		logger.debug(f"Atom (level {index}): Direct result: {direct_result}")

		decompose_args = {'contexts': contexts} if self.module_name == 'multi-hop' else {}
		# self.decompose might return a string or a dict
		logger.debug(f"Atom (level {index}): Calling decompose...")
		decompose_output = decompose_result if decompose_result else await self.decompose(question, **decompose_args)
		logger.debug(f"Atom (level {index}): Decompose raw output (type {type(decompose_output)}): {str(decompose_output)[:200]}...")

		# --- Process Decompose Result (Handle String or Dict) ---
		decompose_result_dict = None
		raw_decompose_for_log = decompose_output # Keep original for logging if parsing fails
		if isinstance(decompose_output, str):
			try:
				# Attempt to parse if it's a string
				logger.debug(f"Atom (level {index}): Attempting to parse decompose string output as JSON.")
				decompose_result_dict = json.loads(decompose_output)
				logger.debug(f"Atom (level {index}): Successfully parsed decompose JSON string.")
			except json.JSONDecodeError as e:
				error_msg = f'JSON parsing failed for decompose result: {e}'
				logger.error(f"Atom (level {index}): {error_msg}", exc_info=True)
				log[index].update({'error': error_msg, 'raw_decompose': raw_decompose_for_log, 'direct': direct_result})
				# Fallback to direct result if JSON is invalid
				logger.warning(f"Atom (level {index}): Falling back to direct result due to decompose JSON parsing error.")
				final_result = {
					'method': 'direct_fallback',
					'response': direct_result.get('response'),
					'answer': direct_result.get('answer'),
				}
				return final_result, log
		elif isinstance(decompose_output, dict):
			# Use directly if it's already a dictionary
			logger.debug(f"Atom (level {index}): Decompose output is already a dict.")
			decompose_result_dict = decompose_output
		else:
			# Handle unexpected type
			error_msg = f'Unexpected type from decompose: {type(decompose_output)}'
			logger.error(f"Atom (level {index}): {error_msg}")
			log[index].update({'error': error_msg, 'raw_decompose': raw_decompose_for_log, 'direct': direct_result})
			logger.warning(f"Atom (level {index}): Falling back to direct result due to unexpected decompose type.")
			final_result = {
				'method': 'direct_fallback',
				'response': direct_result.get('response'),
				'answer': direct_result.get('answer'),
			}
			return final_result, log


		# Handle potential failure in decompose or missing 'sub-questions' key after processing
		if not decompose_result_dict or 'sub-questions' not in decompose_result_dict:
			error_msg = 'Decomposition failed or missing sub-questions'
			logger.error(f"Atom (level {index}): {error_msg}")
			log[index].update({'error': error_msg, 'raw_decompose': raw_decompose_for_log, 'parsed_decompose': decompose_result_dict, 'direct': direct_result})
			# Return direct result if decompose fails or structure is wrong
			logger.warning(f"Atom (level {index}): Falling back to direct result due to decompose failure or missing sub-questions.")
			final_result = {
				'method': 'direct_fallback',
				'response': direct_result.get('response'),
				'answer': direct_result.get('answer'),
			}
			return final_result, log
		logger.debug(f"Atom (level {index}): Processed decompose result (dict): {decompose_result_dict}")

		# --- Step 2: Set recursion depth ---
		logger.debug(f"Atom (level {index}): Step 2 - Setting recursion depth.")
		try:
			current_depth = calculate_depth(decompose_result_dict['sub-questions'])
			logger.debug(f"Atom (level {index}): Calculated depth from decompose result: {current_depth}")
		except Exception as e:
			error_msg = f'Depth calculation failed: {e}'
			logger.error(f"Atom (level {index}): {error_msg}", exc_info=True)
			log[index].update({'error': error_msg, 'direct': direct_result, 'decompose': decompose_result_dict})
			# Return direct result if depth calculation fails
			logger.warning(f"Atom (level {index}): Falling back to direct result due to depth calculation error.")
			final_result = {
				'method': 'direct_fallback',
				'response': direct_result.get('response'),
				'answer': direct_result.get('answer'),
			}
			return final_result, log

		# Determine depth for this level: minimum of default, calculated, or passed-in depth
		depth = depth if depth is not None else self.atom_depth
		depth = min(depth, current_depth)
		logger.debug(f"Atom (level {index}): Effective depth for this level set to: {depth}")


		# --- Step 3: Separate sub-questions and perform Merging (Contraction) ---
		logger.debug(f"Atom (level {index}): Step 3 - Separating sub-questions and merging.")
		# --- Log the sub-questions list before processing ---
		sub_questions_list = decompose_result_dict.get('sub-questions', []) # Use .get for safety
		logger.debug(f"Atom (level {index}): Sub-questions list type: {type(sub_questions_list)}, Content (first 500 chars): {str(sub_questions_list)[:500]}")

		# --- Process sub-questions with logging ---
		independent_subqs = []
		dependent_subqs = []
		if isinstance(sub_questions_list, list):
			for i, sub_q in enumerate(sub_questions_list):
				logger.debug(f"Atom (level {index}): Processing sub_q #{i}: Type={type(sub_q)}, Value={str(sub_q)[:200]}")
				try:
					# Check if it's a dictionary before calling .get()
					if isinstance(sub_q, dict):
						if not sub_q.get('depend'):
							independent_subqs.append(sub_q)
						else:
							dependent_subqs.append(sub_q)
					else:
						logger.warning(f"Atom (level {index}): Sub_q #{i} is not a dictionary, skipping dependency check.")
						# Decide how to handle non-dict items. Maybe add to independent? Or skip?
						# For now, let's skip adding it to either list if it's not a dict.
				except Exception as e:
					logger.error(f"Atom (level {index}): Error processing sub_q #{i}: {e}", exc_info=True)
		else:
			logger.error(f"Atom (level {index}): 'sub-questions' is not a list, cannot separate.")


		logger.debug(f"Atom (level {index}): Separated sub-questions. Independent: {len(independent_subqs)}, Dependent: {len(dependent_subqs)}")

		merging_args = {
			'question': question,
			'decompose_result': decompose_result_dict, # Use the processed dictionary
			'independent_subqs': independent_subqs,
			'dependent_subqs': dependent_subqs
		}
		if self.module_name == 'multi-hop':
			merging_args['contexts'] = contexts

		try:
			logger.debug(f"Atom (level {index}): Calling merging...")
			contractd_thought, contractd_question, contraction_result = await self.merging(**merging_args)
			logger.debug(f"Atom (level {index}): Merging successful. Contraction result: {contraction_result}")
		except Exception as e:
			error_msg = f'Merging failed: {e}'
			logger.error(f"Atom (level {index}): {error_msg}", exc_info=True)
			log[index].update({'error': error_msg, 'direct': direct_result, 'decompose': decompose_result})
			# Fallback if merging fails
			logger.warning(f"Atom (level {index}): Falling back to direct result due to merging error.")
			final_result = {
				'method': 'direct_fallback', # Or potentially decompose fallback?
				'response': direct_result.get('response'),
				'answer': direct_result.get('answer'),
			}
			return final_result, log


		# Augment contraction result for logging and potential use
		contraction_result['contraction_thought'] = contractd_thought
		contraction_result['sub-questions'] = independent_subqs + [{
			'description': contractd_question,
			'response': contraction_result.get('response', ''),
			'answer': contraction_result.get('answer', ''),
			'depend': [] # The contracted question depends on the independent ones implicitly
		}]
		logger.debug(f"Atom (level {index}): Augmented contraction result: {contraction_result}")

		# --- Step 4: Ensemble ---
		logger.debug(f"Atom (level {index}): Step 4 - Performing ensemble.")
		ensemble_args = [question]
		# Ensure responses exist before appending
		responses_for_ensemble = [
			res.get('response') for res in [direct_result, decompose_result, contraction_result] if res and res.get('response')
		]
		ensemble_args.append(responses_for_ensemble)
		logger.debug(f"Atom (level {index}): Ensemble responses count: {len(responses_for_ensemble)}")

		if self.module_name == 'multi-hop':
			ensemble_args.append(contexts)

		try:
			logger.debug(f"Atom (level {index}): Calling ensemble...")
			ensemble_result = await self.ensemble(*ensemble_args)
			ensemble_answer = ensemble_result.get('answer', '')
			logger.debug(f"Atom (level {index}): Ensemble successful. Result: {ensemble_result}")
		except Exception as e:
			error_msg = f'Ensemble failed: {e}'
			logger.error(f"Atom (level {index}): {error_msg}", exc_info=True)
			log[index].update({'error': error_msg, 'direct': direct_result, 'decompose': decompose_result, 'contract': contraction_result})
			# Fallback if ensemble fails - maybe pick best of the three? For now, direct.
			logger.warning(f"Atom (level {index}): Falling back to direct answer due to ensemble error.")
			ensemble_result = {'answer': direct_result.get('answer')} # Use direct answer as fallback
			ensemble_answer = ensemble_result['answer']
			# Consider logging the fallback choice
			logger.debug(f"Atom (level {index}): Using fallback ensemble answer: {ensemble_answer}")


		# --- Step 5: Scoring ---
		logger.debug(f"Atom (level {index}): Step 5 - Scoring results against ensemble answer: '{ensemble_answer}'")
		scores = []
		results_to_score = [direct_result, decompose_result, contraction_result]
		# Check if all valid results have the same answer as the ensemble
		valid_answers = [res.get('answer') for res in results_to_score if res and 'answer' in res]

		if valid_answers and all(ans == ensemble_answer for ans in valid_answers):
			scores = [1.0] * len(results_to_score) # Assign perfect score if all match ensemble
			logger.debug(f"Atom (level {index}): All valid answers matched ensemble. Scores: {scores}")
		else:
			logger.debug(f"Atom (level {index}): Scoring each result individually.")
			for i, result in enumerate(results_to_score):
				method_name = ['direct', 'decompose', 'contract'][i]
				if result and 'answer' in result and self.score_func:
					try:
						# Ensure score_func handles potential None/empty answers gracefully
						result_answer = result.get('answer')
						score_value = self.score_func(result_answer, ensemble_answer)
						scores.append(score_value)
						logger.debug(f"Atom (level {index}): Score for {method_name} ('{result_answer}') vs ensemble ('{ensemble_answer}') = {score_value}")
					except Exception as e:
						logger.error(f"Atom (level {index}): Scoring failed for {method_name} result: {e}", exc_info=True) # Replaced print with logger.error
						scores.append(0.0) # Assign 0 score on error
				else:
					logger.debug(f"Atom (level {index}): Assigning 0 score to {method_name} (invalid result, missing answer, or no score_func).")
					scores.append(0.0) # Assign 0 score if result is invalid or missing answer

		# Ensure scores list matches the number of results
		while len(scores) < len(results_to_score):
			scores.append(0.0)
		logger.debug(f"Atom (level {index}): Final scores: {scores}")

		# --- Step 6: Update Log ---
		logger.debug(f"Atom (level {index}): Step 6 - Updating log dictionary.")
		log[index].update({
			'scores': scores,
			'direct': direct_result,
			'decompose': decompose_result,
			'contract': contraction_result,
			'ensemble': ensemble_result # Log ensemble result too
		})

		# --- Step 7: Select Best Method ---
		logger.debug(f"Atom (level {index}): Step 7 - Selecting best method based on scores.")
		methods = {
			0: ('direct', direct_result),
			1: ('decompose', decompose_result),
			2: ('contract', contraction_result),
			-1: ('ensemble', ensemble_result) # Fallback/default
		}

		best_method_idx = -1
		if scores: # Check if scores list is not empty
			max_score = max(scores)
			# Find the first index matching the max score
			try:
				best_method_idx = scores.index(max_score)
			except ValueError:
				logger.warning(f"Atom (level {index}): Max score not found in scores list {scores}. Defaulting to ensemble.")
				best_method_idx = -1 # Should not happen if scores is not empty, but safety first

		# If multiple methods have the same max score, default to ensemble or a predefined order?
		# Current logic takes the first one found (direct > decompose > contract).
		# If max score is low (e.g., 0), maybe ensemble is better? Add threshold?
		# For now, stick to the index of max score, defaulting to ensemble if index is invalid.

		method, result = methods.get(best_method_idx, methods[-1])
		log[index]['method'] = method
		logger.info(f"Atom (level {index}): Selected best method: {method} (Index: {best_method_idx}, Score: {scores[best_method_idx] if best_method_idx != -1 else 'N/A - Ensemble Fallback'})")

		# --- Step 8: Recursive Call or Return ---
		logger.debug(f"Atom (level {index}): Step 8 - Preparing return value.")
		# Decide if recursion is needed based on selected method and depth
		# Currently, recursion isn't implemented in the original logic based on method choice.
		# The original logic seems to run one level and return the best.
		# If recursive refinement was intended, it would go here, passing the chosen 'result'
		# and decrementing 'depth'.

		# Return the selected result for this level
		# The final result structure is slightly different for the top level (index == 0)
		if index == 0:
			final_return_result = {
				'method': method,
				'response': result.get('response'),
				'answer': result.get('answer'),
			}
			logger.info(f"Atom finished (Top Level). Returning method: {method}, Final Result: {final_return_result}")
			return final_return_result, log
		else:
			# For recursive calls, just return the result dictionary
			logger.info(f"Atom finished (Recursive Level {index}). Returning method: {method}, Result: {result}")
			return result, log


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
