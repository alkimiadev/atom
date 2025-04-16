import asyncio
from functools import wraps
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

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
		self.module_name = module_name
		if module_name == 'math':
			self.prompter = math
			self.score_func = score_math
		elif module_name == 'multi-choice':
			self.prompter = multichoice
			self.score_func = score_mc
		elif module_name == 'multi-hop':
			self.prompter = multihop
			self.score_func = score_mh
		else:
			# Consider raising an error for unknown module types
			print(f'Warning: Unknown module type configured: {module_name}')
			self.prompter = None
			self.score_func = None

	# --- Decorator and Context Manager ---

	def retry(self, func_name):
		'''Decorator to retry LLM calls with checks.'''
		def decorator(func):
			@wraps(func)
			async def wrapper(*args, **kwargs):
				# Note: 'self' is the first arg because it's bound from the method call
				instance = args[0]
				actual_args = args[1:] # Exclude self for prompter call

				retries = instance.max_retries
				last_result = {} # Keep track of the last result in case all retries fail

				while retries >= 0:
					if not instance.prompter:
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

					# Extract result based on format
					if response_format == 'json_object':
						result = extract_json(response)
					else: # text format
						result = extract_xml(response)

					# Store raw response if result is a dict
					if isinstance(result, dict):
						result['response'] = response
						last_result = result # Update last successful-ish result
					else:
						last_result = {'response': response, 'error': 'Extraction failed'}


					# Check if the result is valid using the prompter's check function
					if instance.prompter.check(func_name, result):
						return result # Success

					retries -= 1 # Decrement retries on failure

				# --- All retries failed ---
				if instance.max_retries > 1: # Only count failures if retries were enabled
					instance.failure_count += 1
				if instance.failure_count > 300: # Safety break
					raise Exception('Too many failures across multiple calls')

				# Return the last result obtained, even if it failed the check
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
		retries = self.label_retries
		if self.module_name == 'multi-hop':
			if 'contexts' not in kwargs:
				raise Exception('Multi-hop must have contexts')
			contexts = kwargs['contexts']
			multistep_result = await self.multistep(question, contexts)
			label_result = {}
			while retries > 0:
				# Pass the dictionary directly to label for multi-hop
				label_result = await self.label(question, multistep_result)
				try:
					# Check if lengths match before accessing keys
					if 'sub-questions' not in label_result or 'sub-questions' not in multistep_result:
						raise ValueError("Missing 'sub-questions' key in results")
					if len(label_result.get('sub-questions', [])) != len(multistep_result.get('sub-questions', [])):
						retries -= 1
						continue
					calculate_depth(label_result['sub-questions']) # Check dependencies are valid
					break # Success
				except Exception as e:
					print(f"Labeling/Depth calculation failed (attempt {self.label_retries - retries + 1}): {e}")
					retries -= 1
					if retries == 0:
						print("Max label retries reached for multi-hop.")
						# Return multistep result if labeling fails completely? Or raise?
						# For now, return the potentially incomplete label_result or empty if error
						return label_result if isinstance(label_result, dict) else {}
					await asyncio.sleep(1) # Small delay before retrying label
					continue

			# Combine results if successful
			if 'sub-questions' in label_result and 'sub-questions' in multistep_result:
				for step, note in zip(multistep_result['sub-questions'], label_result['sub-questions']):
					step['depend'] = note.get('depend', []) # Use .get for safety
			return multistep_result

		else: # Math or Multi-choice
			multistep_result = await self.multistep(question)
			result = {}
			while retries > 0:
				# Pass response and answer strings to label
				result = await self.label(question, multistep_result.get('response', ''), multistep_result.get('answer', ''))
				try:
					if 'sub-questions' not in result:
						raise ValueError("Missing 'sub-questions' key in label result")
					calculate_depth(result['sub-questions'])
					result['response'] = multistep_result.get('response', '') # Add original response back
					break # Success
				except Exception as e:
					print(f"Labeling/Depth calculation failed (attempt {self.label_retries - retries + 1}): {e}")
					retries -= 1
					if retries == 0:
						print("Max label retries reached.")
						return result if isinstance(result, dict) else {} # Return last attempt
					await asyncio.sleep(1)
					continue
			return result

	async def merging(self, question: str, decompose_result: dict, independent_subqs: list, dependent_subqs: list, **kwargs) -> Tuple[str, str, Dict[str, Any]]:
		'''Merges independent and dependent sub-questions via contraction.'''
		contract_args = [question, decompose_result, independent_subqs, dependent_subqs]
		if self.module_name == 'multi-hop':
			contract_args.append(kwargs.get('contexts')) # Use .get for safety

		contractd_result = await self.contract(*contract_args)

		# Extract thought process and optimized question
		contractd_thought = contractd_result.get('response', '')
		contractd_question = contractd_result.get('question', '')

		# Solve the optimized question
		direct_args = [contractd_question]
		if self.module_name == 'multi-hop':
			# Pass context from contract result or original contexts
			direct_args.append(contractd_result.get('context', kwargs.get('contexts')))

		contraction_result = await self.direct(*direct_args)

		return contractd_thought, contractd_question, contraction_result

	async def atom(self, question: str, contexts: Optional[str] = None, direct_result: Optional[Dict] = None, decompose_result: Optional[Dict] = None, depth: Optional[int] = None, log: Optional[Dict] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
		'''Performs the Atom of Thoughts process recursively.'''
		# Initialize logging for this level
		log = log if log is not None else {}
		index = len(log)
		log[index] = {}

		# Base case: depth limit reached
		if depth == 0:
			# Return format consistent with recursive step but indicate termination
			return {'method': 'depth_limit', 'response': None, 'answer': None}, log

		# --- Step 1: Get results from Direct and Decompose approaches ---
		direct_args = [question]
		if self.module_name == 'multi-hop':
			direct_args.append(contexts)
		# Use provided results if available (for recursion), otherwise compute
		direct_result = direct_result if direct_result else await self.direct(*direct_args)

		decompose_args = {'contexts': contexts} if self.module_name == 'multi-hop' else {}
		decompose_result = decompose_result if decompose_result else await self.decompose(question, **decompose_args)

		# Handle potential failure in decompose
		if not decompose_result or 'sub-questions' not in decompose_result:
			log[index].update({'error': 'Decomposition failed', 'direct': direct_result})
			# Return direct result if decompose fails
			final_result = {
				'method': 'direct_fallback',
				'response': direct_result.get('response'),
				'answer': direct_result.get('answer'),
			}
			return final_result, log

		# --- Step 2: Set recursion depth ---
		try:
			current_depth = calculate_depth(decompose_result['sub-questions'])
		except Exception as e:
			log[index].update({'error': f'Depth calculation failed: {e}', 'direct': direct_result, 'decompose': decompose_result})
			# Return direct result if depth calculation fails
			final_result = {
				'method': 'direct_fallback',
				'response': direct_result.get('response'),
				'answer': direct_result.get('answer'),
			}
			return final_result, log

		# Determine depth for this level: minimum of default, calculated, or passed-in depth
		depth = depth if depth is not None else self.atom_depth
		depth = min(depth, current_depth)


		# --- Step 3: Separate sub-questions and perform Merging (Contraction) ---
		independent_subqs = [sub_q for sub_q in decompose_result['sub-questions'] if not sub_q.get('depend')]
		dependent_subqs = [sub_q for sub_q in decompose_result['sub-questions'] if sub_q.get('depend')]

		merging_args = {
			'question': question,
			'decompose_result': decompose_result,
			'independent_subqs': independent_subqs,
			'dependent_subqs': dependent_subqs
		}
		if self.module_name == 'multi-hop':
			merging_args['contexts'] = contexts

		try:
			contractd_thought, contractd_question, contraction_result = await self.merging(**merging_args)
		except Exception as e:
			log[index].update({'error': f'Merging failed: {e}', 'direct': direct_result, 'decompose': decompose_result})
			# Fallback if merging fails
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

		# --- Step 4: Ensemble ---
		ensemble_args = [question]
		# Ensure responses exist before appending
		responses_for_ensemble = [
			res.get('response') for res in [direct_result, decompose_result, contraction_result] if res and res.get('response')
		]
		ensemble_args.append(responses_for_ensemble)

		if self.module_name == 'multi-hop':
			ensemble_args.append(contexts)

		try:
			ensemble_result = await self.ensemble(*ensemble_args)
			ensemble_answer = ensemble_result.get('answer', '')
		except Exception as e:
			log[index].update({'error': f'Ensemble failed: {e}', 'direct': direct_result, 'decompose': decompose_result, 'contract': contraction_result})
			# Fallback if ensemble fails - maybe pick best of the three? For now, direct.
			ensemble_result = {'answer': direct_result.get('answer')} # Use direct answer as fallback
			ensemble_answer = ensemble_result['answer']
			# Consider logging the fallback choice


		# --- Step 5: Scoring ---
		scores = []
		results_to_score = [direct_result, decompose_result, contraction_result]
		# Check if all valid results have the same answer as the ensemble
		valid_answers = [res.get('answer') for res in results_to_score if res and 'answer' in res]

		if valid_answers and all(ans == ensemble_answer for ans in valid_answers):
			scores = [1.0] * len(results_to_score) # Assign perfect score if all match ensemble
		else:
			for result in results_to_score:
				if result and 'answer' in result and self.score_func:
					try:
						# Ensure score_func handles potential None/empty answers gracefully
						score_value = self.score_func(result.get('answer'), ensemble_answer)
						scores.append(score_value)
					except Exception as e:
						print(f"Scoring failed for a result: {e}")
						scores.append(0.0) # Assign 0 score on error
				else:
					scores.append(0.0) # Assign 0 score if result is invalid or missing answer

		# Ensure scores list matches the number of results
		while len(scores) < len(results_to_score):
			scores.append(0.0)

		# --- Step 6: Update Log ---
		log[index].update({
			'scores': scores,
			'direct': direct_result,
			'decompose': decompose_result,
			'contract': contraction_result,
			'ensemble': ensemble_result # Log ensemble result too
		})

		# --- Step 7: Select Best Method ---
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
				best_method_idx = -1 # Should not happen if scores is not empty, but safety first

		# If multiple methods have the same max score, default to ensemble or a predefined order?
		# Current logic takes the first one found (direct > decompose > contract).
		# If max score is low (e.g., 0), maybe ensemble is better? Add threshold?
		# For now, stick to the index of max score, defaulting to ensemble if index is invalid.

		method, result = methods.get(best_method_idx, methods[-1])
		log[index]['method'] = method

		# --- Step 8: Recursive Call or Return ---
		# Decide if recursion is needed based on selected method and depth
		# Currently, recursion isn't implemented in the original logic based on method choice.
		# The original logic seems to run one level and return the best.
		# If recursive refinement was intended, it would go here, passing the chosen 'result'
		# and decrementing 'depth'.

		# Return the selected result for this level
		# The final result structure is slightly different for the top level (index == 0)
		if index == 0:
			return {
				'method': method,
				'response': result.get('response'),
				'answer': result.get('answer'),
			}, log
		else:
			# For recursive calls, just return the result dictionary
			return result, log


	async def plugin(self, question: str, contexts: Optional[str] = None, sample_num: int = 3) -> str:
		'''Generates multiple decompositions and selects the best contracted question.'''

		async def process_sample():
			'''Helper function to process one sample decomposition and contraction.'''
			try:
				# Get decompose result
				decompose_args = {'contexts': contexts} if self.module_name == 'multi-hop' else {}
				decompose_result = await self.decompose(question, **decompose_args)

				if not decompose_result or 'sub-questions' not in decompose_result:
					print("Plugin: Decomposition failed for a sample.")
					return None # Indicate failure for this sample

				# Separate independent and dependent sub-questions
				independent_subqs = [sub_q for sub_q in decompose_result['sub-questions'] if not sub_q.get('depend')]
				dependent_subqs = [sub_q for sub_q in decompose_result['sub-questions'] if sub_q.get('depend')]

				# Get contraction result
				merging_args = {
					'question': question,
					'decompose_result': decompose_result,
					'independent_subqs': independent_subqs,
					'dependent_subqs': dependent_subqs
				}
				if self.module_name == 'multi-hop':
					merging_args['contexts'] = contexts

				contractd_thought, contractd_question, contraction_result = await self.merging(**merging_args)

				# Estimate token count (simple split for now)
				token_count = len(contraction_result.get('response', '').split())

				return {
					# 'decompose_result': decompose_result, # Not needed for final selection
					'contractd_thought': contractd_thought,
					'contractd_question': contractd_question,
					'contraction_result': contraction_result,
					'token_count': token_count
				}
			except Exception as e:
				print(f"Plugin: Error processing sample: {e}")
				return None # Indicate failure

		# Execute all samples in parallel
		tasks = [process_sample() for _ in range(sample_num)]
		all_results = await asyncio.gather(*tasks)
		valid_results = [r for r in all_results if r is not None] # Filter out failed samples

		if not valid_results:
			print("Plugin: All samples failed processing.")
			# Fallback: return original question? Or raise error?
			return question # Return original question as fallback

		# Get direct result for original question (used for ensemble scoring)
		direct_args = [question]
		if self.module_name == 'multi-hop':
			direct_args.append(contexts)
		direct_result = await self.direct(*direct_args)

		# Get ensemble result from all valid contracted results + direct result
		all_responses = [direct_result.get('response', '')] + \
						[r['contraction_result'].get('response', '') for r in valid_results]
		ensemble_args = [question, [resp for resp in all_responses if resp]] # Filter empty responses
		if self.module_name == 'multi-hop':
			ensemble_args.append(contexts)

		try:
			ensemble_result = await self.ensemble(*ensemble_args)
			ensemble_answer = ensemble_result.get('answer', '')
		except Exception as e:
			print(f"Plugin: Ensemble failed: {e}. Using direct answer as fallback.")
			ensemble_answer = direct_result.get('answer', '') # Fallback scoring reference


		# Calculate scores for each valid contracted result vs ensemble
		scores = []
		for result in valid_results:
			contraction_answer = result['contraction_result'].get('answer')
			if contraction_answer is not None and self.score_func:
				try:
					scores.append(self.score_func(contraction_answer, ensemble_answer))
				except Exception as e:
					print(f"Plugin: Scoring failed for a result: {e}")
					scores.append(0.0)
			else:
				scores.append(0.0) # Score 0 if answer missing or no score func

		# Find the best result(s) - those with the highest score
		if not scores:
			print("Plugin: No valid scores calculated.")
			# Fallback: return first valid result's question or original?
			return valid_results[0]['contractd_question'] if valid_results else question

		max_score = max(scores)
		best_indices = [i for i, s in enumerate(scores) if s == max_score]

		# Among the best results, find the one with the lowest token count
		best_index_in_valid = min(best_indices, key=lambda i: valid_results[i]['token_count'])

		# Return the best contracted question
		best_result = valid_results[best_index_in_valid]
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