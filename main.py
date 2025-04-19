import asyncio
import os
import time
import argparse
import logging # Added import
import datetime
import random # Added import for sampling
from math import ceil # Added for batch calculation
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional # Added Optional
from tqdm.asyncio import tqdm

from experiment.dataset import load_data
from experiment.processor import AtomProcessor # Added
# Removed set_module, atom, plugin from below import
from experiment.utils import (
	duration_formatter,
	load_json,
	save_json,
	get_next_log_file,
	get_file_count,
)
from llm import LLMManager # Import the class

# Configuration constants
LOG_DIR = "log/{dataset}/{size}"
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s'
LOG_LEVEL = logging.INFO # Default level, can be adjusted

# Dataset configuration
@dataclass
class DatasetConfig:
	question_key: str
	answer_key: str
	module_type: str
	scoring_function: str
	
	def requires_context(self) -> bool:
		return self.module_type == "multi-hop"

# Dataset configuration mapping
DATASET_CONFIGS = {
	"gsm8k": DatasetConfig(question_key="question", answer_key="answer", 
						  module_type="math", scoring_function="score_math"),
	"math": DatasetConfig(question_key="problem", answer_key="solution", 
						 module_type="math", scoring_function="score_math"),
	"bbh": DatasetConfig(question_key="input", answer_key="target", 
						module_type="multi-choice", scoring_function="score_mc"),
	"mmlu": DatasetConfig(question_key=["Question", "A", "B", "C", "D"], answer_key="Answer", 
						 module_type="multi-choice", scoring_function="score_mc"),
	"hotpotqa": DatasetConfig(question_key="question", answer_key="answer", 
							 module_type="multi-hop", scoring_function="score_mh"),
	"longbench": DatasetConfig(question_key="input", answer_key="answers", 
							  module_type="multi-hop", scoring_function="score_mh"),
}


# --- Logging Setup ---
def setup_logging(dataset: str, interval: str, log_dir_format: str):
	"""Configures logging to file and console."""
	log_directory = log_dir_format.format(dataset=dataset, size=interval)
	os.makedirs(log_directory, exist_ok=True)

	timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
	log_filename = os.path.join(log_directory, f'run_{timestamp}.log')

	# Remove existing handlers if reconfiguring
	for handler in logging.root.handlers[:]:
		logging.root.removeHandler(handler)

	logging.basicConfig(
		level=LOG_LEVEL,
		format=LOG_FORMAT,
		handlers=[
			logging.FileHandler(log_filename),
			logging.StreamHandler() # Log to console as well
		]
	)
	logging.info(f"Logging configured. Log file: {log_filename}")

# --- Experiment Runner ---
logger = logging.getLogger(__name__) # Add logger for this module

class ExperimentRunner:
	def __init__(self, dataset: str, model: str, start: int = 0, end: int = -1, mode: str = "atom",
				 batch_size: Optional[int] = None, sample_size: Optional[int] = None):
		# Initialize experiment runner
		logger.info(f"Initializing ExperimentRunner: dataset={dataset}, model={model}, start={start}, end={end}, mode={mode}, batch_size={batch_size}, sample_size={sample_size}")
		self.dataset = dataset
		self.start = start
		self.end = None if end == -1 else end
		self.mode = mode
		self.batch_size = batch_size
		self.sample_size = sample_size
		self.timestamp = time.time()

		# Determine interval string for logging/output dirs
		if self.sample_size is not None:
			self.interval = f"sample-{self.sample_size}"
		elif self.end is None:
			self.interval = "full"
		else:
			self.interval = f"{start}-{end}"
		if self.batch_size is not None and self.sample_size is None: # Add batch info if batching a slice/full
			self.interval += f"_batch-{self.batch_size}"

		# Validate dataset support
		if dataset not in DATASET_CONFIGS:
			logger.error(f"Unsupported dataset: {dataset}") # Log error
			raise ValueError(f"Unsupported dataset: {dataset}")

		self.config = DATASET_CONFIGS[dataset]
		self.llm_manager = LLMManager() # Instantiate LLMManager
		logger.info(f"Setting LLM model to: {model}") # Log model setting
		self.llm_manager.set_model(model) # Use manager method
		# Pass the llm_manager to the processor (requires AtomProcessor update)
		self.processor = AtomProcessor(llm_manager=self.llm_manager)

	async def gather_results(self, testset: List[Dict[str, Any]]) -> List[Any]:
		# Collect experiment results
		logger.info(f"Configuring AtomProcessor for module: {self.config.module_type}")
		self.processor.configure_module(self.config.module_type) # Use processor method

		question_key = self.config.question_key
		tasks = []
		logger.debug(f"Preparing tasks for dataset '{self.dataset}' with question key(s): {question_key}") # Log task prep

		if self.config.requires_context():
			from experiment.prompter.multihop import contexts
			# Handle case where question_key is a list
			if isinstance(question_key, list):
				formatted_questions = [self._format_question_from_keys(item, question_key) for item in testset]
				tasks = [self.processor.atom(question, contexts(item, self.dataset)) # Use processor method
						 for question, item in zip(formatted_questions, testset)]
			else:
				tasks = [self.processor.atom(item[question_key], contexts(item, self.dataset)) for item in testset] # Use processor method
		else:
			# Handle case where question_key is a list
			if isinstance(question_key, list):
				tasks = [self.processor.atom(self._format_question_from_keys(item, question_key)) for item in testset] # Use processor method
			else:
				tasks = [self.processor.atom(item[question_key]) for item in testset] # Use processor method

		return await tqdm.gather(*tasks, desc=f"Processing {self.dataset} tasks")
	
	def _format_question_from_keys(self, item: Dict[str, Any], keys: List[str]) -> str:
		# When question_key is a list, concatenate values from multiple keys into a single question
		parts = []
		for key in keys:
			if key in item:
				parts.append(f"{key}: {item[key]}")
		return "\n".join(parts)
	
	def construct_entry(self, result: Tuple[Dict[str, Any], Any], data: Dict[str, Any]) -> Dict[str, Any]:
		# Construct result entry
		logger.debug(f"Constructing entry. Raw result: {result}, Raw data: {data}") # ADDED LOG
		result_data, log = result
		question_key = self.config.question_key
		answer_key = self.config.answer_key
		
		# Handle case where question_key is a list
		if isinstance(question_key, list):
			question = self._format_question_from_keys(data, question_key)
		else:
			question = data[question_key]
			
		groundtruth = data[answer_key]
		
		entry = {
			"problem": question,
			"groundtruth": groundtruth,
			"response": result_data.get("response"),
			"answer": result_data.get("answer"),
			"log": log
		}
		
		# Dynamically import scoring function
		scoring_function = getattr(__import__(f"experiment.utils", fromlist=[self.config.scoring_function]), 
								  self.config.scoring_function)
		
		# Log values just before scoring
		logger.debug(f"Values before scoring: answer='{entry['answer']}', groundtruth='{entry['groundtruth']}'") # ADDED LOG
		
		# Check if answer is None before scoring
		if entry["answer"] is None:
			logger.warning(f"Prediction (answer) is None for question: {entry['problem'][:100]}... Assigning score 0.")
			entry["score"] = 0
		else:
			# Pass different parameters based on scoring function
			if self.config.scoring_function == "score_math":
				entry["score"] = scoring_function(entry["answer"], groundtruth, self.dataset)
			else:
				entry["score"] = scoring_function(entry["answer"], groundtruth)
		return entry
	
	def update_score_log(self, accuracy: float, num_items_processed: int) -> None:
		# Update score log
		log_entry = {
			"mode": self.mode,
			"model": self.llm_manager.model_name, # Log the model used
			"dataset": self.dataset,
			"interval": self.interval, # Use the potentially modified interval string
			"start": self.start if self.sample_size is None else None, # Log start/end only if not sampling
			"end": self.end if self.sample_size is None else None,
			"sample_size": self.sample_size,
			"batch_size": self.batch_size,
			"num_items_processed": num_items_processed,
			# Use manager methods for stats
			"token": {"prompt": self.llm_manager.get_token()[0], "completion": self.llm_manager.get_token()[1]},
			"call_count": self.llm_manager.get_call_count(),
			"accuracy": accuracy,
			"timestamp": datetime.datetime.now().isoformat(), # Add timestamp of score logging
			"duration_seconds": time.time() - self.timestamp, # Add duration
		}

		score_log_file = LOG_DIR.format(dataset=self.dataset, size=self.interval) + "/score.json"
		# Ensure directory exists before trying to load/save
		os.makedirs(os.path.dirname(score_log_file), exist_ok=True)
		existing_log = load_json(score_log_file) if os.path.exists(score_log_file) else {}
		count = get_file_count(LOG_DIR, self.interval, self.dataset, exclude_score=True)

		if self.dataset not in existing_log:
			existing_log[self.dataset] = {}
		existing_log[self.dataset][str(count)] = log_entry
		save_json(score_log_file, existing_log)
	
	async def run(self) -> float:
		# Run experiment and return accuracy
		run_description = f"mode={self.mode}, dataset={self.dataset}, interval={self.interval}"
		if self.sample_size is not None:
			run_description += f", sample_size={self.sample_size}"
		elif self.batch_size is not None:
			run_description += f", batch_size={self.batch_size}"
		logger.info(f"Starting experiment run: {run_description}")
		print(f"Running {self.mode} experiment on {self.dataset} dataset ({self.interval})") # Keep print for user visibility

		# --- Load/Select Data ---
		if self.sample_size is not None:
			logger.info(f"Loading full test data for {self.dataset} to sample {self.sample_size} items.")
			full_testset = load_data(self.dataset, "test")
			if self.sample_size > len(full_testset):
				logger.warning(f"Sample size ({self.sample_size}) is larger than the dataset size ({len(full_testset)}). Using the entire dataset.")
				self.sample_size = len(full_testset)
			testset = random.sample(full_testset, self.sample_size)
			logger.info(f"Sampled {len(testset)} items.")
		else:
			start_idx = self.start
			end_idx = self.end
			logger.info(f"Loading test data for {self.dataset} [{start_idx}:{end_idx}]")
			# Load the slice; handle potential slicing issues if end is None
			full_testset = load_data(self.dataset, "test")
			testset = full_testset[start_idx:end_idx] # end_idx=None handles slicing to the end
			logger.info(f"Loaded {len(testset)} items.")

		if not testset:
			logger.error("Test set is empty. Check start/end indices or sampling parameters.")
			print("Error: Test set is empty. Aborting run.")
			return 0.0

		# --- Process Data (Batching or Full) ---
		all_results = []
		all_json_obj = []
		original_indices = list(range(self.start, self.start + len(testset))) if self.sample_size is None else ["sampled"] * len(testset) # Track original index if not sampling

		if self.batch_size is not None and self.sample_size is None:
			num_batches = ceil(len(testset) / self.batch_size)
			logger.info(f"Processing {len(testset)} items in {num_batches} batches of size {self.batch_size}.")
			for i in range(0, len(testset), self.batch_size):
				batch_start_index = i
				batch_end_index = min(i + self.batch_size, len(testset))
				batch_data = testset[batch_start_index:batch_end_index]
				batch_original_indices = original_indices[batch_start_index:batch_end_index]
				batch_num = (i // self.batch_size) + 1
				logger.info(f"Processing batch {batch_num}/{num_batches} (Items {self.start + batch_start_index}-{self.start + batch_end_index -1})...")

				batch_results = await self.gather_results(batch_data)
				logger.info(f"Gathered {len(batch_results)} results for batch {batch_num}.")
				all_results.extend(batch_results) # Append raw results if needed later

				# Construct entries for the current batch
				logger.info(f"Constructing result entries for batch {batch_num}...")
				for j, (result, data) in enumerate(zip(batch_results, batch_data)):
					item_original_index = batch_original_indices[j]
					try:
						entry = self.construct_entry(result, data)
						entry["original_index"] = item_original_index # Add original index
						all_json_obj.append(entry)
					except Exception as e:
						log_index = f"sampled_{j}" if self.sample_size is not None else self.start + batch_start_index + j
						logger.error(f"Error constructing entry for item {log_index}: {e}", exc_info=True)
						all_json_obj.append({
							"problem": "Error processing item",
							"groundtruth": data.get(self.config.answer_key, "N/A"),
							"response": None,
							"answer": None,
							"log": result[1] if isinstance(result, tuple) and len(result) > 1 else None,
							"score": 0,
							"error": str(e),
							"original_index": item_original_index
						})
			logger.info("Finished processing all batches.")
		else:
			# Process all at once (sampling or no batching)
			logger.info(f"Processing {len(testset)} items all at once...")
			results = await self.gather_results(testset)
			logger.info(f"Gathered {len(results)} results.")
			all_results = results # Assign results directly

			# Build results
			logger.info("Constructing result entries...")
			for i, (result, data) in enumerate(zip(results, testset)):
				item_original_index = original_indices[i]
				try:
					entry = self.construct_entry(result, data)
					entry["original_index"] = item_original_index # Add original index
					all_json_obj.append(entry)
				except Exception as e:
					log_index = f"sampled_{i}" if self.sample_size is not None else self.start + i
					logger.error(f"Error constructing entry for item {log_index}: {e}", exc_info=True)
					all_json_obj.append({
						"problem": "Error processing item",
						"groundtruth": data.get(self.config.answer_key, "N/A"),
						"response": None,
						"answer": None,
						"log": result[1] if isinstance(result, tuple) and len(result) > 1 else None,
						"score": 0,
						"error": str(e),
						"original_index": item_original_index
					})

		# --- Calculate Accuracy ---

		if not all_json_obj:
			logger.error("No results were successfully processed.") # Log if no results
			accuracy = 0.0
		else:
			accuracy = sum(entry["score"] for entry in all_json_obj) / len(all_json_obj)
		logger.info(f"Calculated final accuracy: {accuracy:.4f} across {len(all_json_obj)} items.") # Log accuracy

		# Save results
		log_file = get_next_log_file(LOG_DIR, self.interval, self.dataset, exclude_score=True) # Pass exclude_score
		logger.info(f"Saving detailed results to: {log_file}") # Log save path
		save_json(log_file, all_json_obj)

		# Update score log
		logger.info("Updating score log.") # Log score update
		self.update_score_log(accuracy, len(all_json_obj))

		# Print result summary (Keep for user visibility)
		print(f"Items processed: {len(all_json_obj)}")
		print(f"Accuracy: {accuracy:.4f}")
		print(f"Time taken: {duration_formatter(time.time() - self.timestamp)}")
		logger.info(f"Experiment run finished for interval: {self.interval}") # Log run finish

		return accuracy


async def optimize_dataset(dataset: str, model: str, start: int = 0, end: int = -1):
	# Optimize dataset questions and save to new file
	logger.info(f"Starting dataset optimization: dataset={dataset}, model={model}, start={start}, end={end}")
	print(f"Optimizing {dataset} dataset questions from index {start} to {end}") # Keep print
	timestamp = time.time()
	
	# Instantiate LLMManager and set model
	llm_manager = LLMManager()
	logger.info(f"Setting LLM model to: {model}")
	llm_manager.set_model(model) # Use manager method
	# Pass the llm_manager to the processor (requires AtomProcessor update)
	processor = AtomProcessor(llm_manager=llm_manager)
	config = DATASET_CONFIGS[dataset]
	logger.info(f"Configuring AtomProcessor for module: {config.module_type}")
	processor.configure_module(config.module_type) # Use processor method

	# Load test set
	logger.info(f"Loading test data for {dataset} [{start}:{end}]")
	testset = load_data(dataset, "test")[start:None if end == -1 else end]
	logger.info(f"Loaded {len(testset)} items for optimization.") # Log item count
	question_key = config.question_key
	if isinstance(question_key, list):
		question_key = question_key[0]
	
	# Create tasks
	async def process_item(item):
		try:
			if config.requires_context():
				from experiment.prompter.multihop import contexts
				optimized_question = await processor.plugin(item[question_key], contexts(item, dataset)) # Use processor method
			else:
				optimized_question = await processor.plugin(item[question_key]) # Use processor method

			# Create new entry
			new_item = item.copy()
			new_item["original_question"] = item[question_key]
			new_item[question_key] = optimized_question
			return new_item
		except Exception as e:
			logger.error(f"Error processing item during optimization: {e}", exc_info=True) # Log optimization error
			# Optionally return a modified item indicating error
			item_with_error = item.copy()
			item_with_error["optimization_error"] = str(e)
			return item_with_error # Return modified item on error

	# Process all items in parallel
	logger.info("Starting parallel optimization processing...") # Log parallel start
	tasks = [process_item(item) for item in testset]
	optimized_data = await tqdm.gather(*tasks, desc=f"Optimizing {dataset} questions")
	logger.info(f"Finished optimization processing. Got {len(optimized_data)} results.") # Log parallel end

	# Ensure output directory exists
	output_dir = f"experiment/data/{dataset}"
	logger.info(f"Ensuring output directory exists: {output_dir}") # Log dir check
	os.makedirs(output_dir, exist_ok=True)
	
	# Save optimized dataset
	output_path = f"experiment/data/{dataset}/contracted.json"
	logger.info(f"Saving optimized dataset to: {output_path}") # Log save path
	save_json(output_path, optimized_data)

	elapsed_time = time.time() - timestamp
	logger.info(f"Dataset optimization finished. Time taken: {duration_formatter(elapsed_time)}") # Log optimization finish
	print(f"Optimized dataset saved to {output_path}") # Keep print
	print(f"Time taken: {duration_formatter(elapsed_time)}") # Keep print
	
	return optimized_data

async def main():
	# Main function
	parser = argparse.ArgumentParser(description='Run experiments on various datasets')
	parser.add_argument('--dataset', type=str, default='math', 
						choices=list(DATASET_CONFIGS.keys()),
						help='Dataset to run experiment on')
	parser.add_argument('--start', type=int, default=0, 
						help='Start index of the dataset')
	parser.add_argument('--end', type=int, default=2, 
						help='End index of the dataset (-1 for all)')
	parser.add_argument('--model', type=str, default='gpt-4o-mini',
						help='Model to use for the experiment')
	parser.add_argument('--mode', type=str, choices=['atom', 'plugin'], default='atom',
						help='Mode: atom (standard experiment) or plugin (generate contracted dataset)')
	parser.add_argument('--batch-size', type=int, default=None,
						help='Process dataset in batches of this size (mutually exclusive with --sample-size)')
	parser.add_argument('--sample-size', type=int, default=None,
						help='Randomly sample this many items from the dataset (mutually exclusive with --batch-size)')

	args = parser.parse_args()

	# Validate mutually exclusive arguments
	if args.batch_size is not None and args.sample_size is not None:
		parser.error("Arguments --batch-size and --sample-size are mutually exclusive.")

	# --- Setup Logging ---
	# Determine interval string for logging setup (initial guess, runner might refine)
	if args.sample_size is not None:
		interval = f"sample-{args.sample_size}"
	elif args.end == -1:
		interval = "full"
	else:
		interval = f"{args.start}-{args.end}"
	if args.batch_size is not None and args.sample_size is None:
		interval += f"_batch-{args.batch_size}"

	setup_logging(args.dataset, interval, LOG_DIR) # Call logging setup
	logger.info(f"Parsed arguments: {args}") # Log arguments
	# --- End Logging Setup ---

	if args.mode == 'plugin':
		# Run plugin mode
		logger.info("Starting in plugin (optimization) mode.") # Log mode start
		await optimize_dataset(
			dataset=args.dataset,
			model=args.model,
			start=args.start,
			end=args.end
		)
	elif args.mode == 'atom':
		# Run standard experiment
		logger.info("Starting in atom (standard experiment) mode.") # Log mode start
		runner = ExperimentRunner(
			dataset=args.dataset,
			model=args.model,
			start=args.start,
			end=args.end,
			mode=args.mode,
			batch_size=args.batch_size,
			sample_size=args.sample_size
		)
		await runner.run()
	else:
		logger.error(f"Invalid mode specified: {args.mode}") # Log invalid mode
		raise ValueError(f"Invalid mode: {args.mode}")

if __name__ == "__main__":
	# Wrap the main execution in a try-except block to catch top-level errors
	try:
		asyncio.run(main())
	except Exception as e:
		# Log any uncaught exceptions before exiting
		logging.critical(f"Unhandled exception occurred: {e}", exc_info=True) # Log critical error
		raise # Re-raise the exception after logging
