import asyncio
import os
import time
import argparse
import logging # Added import
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
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
from llm import get_token, get_call_count, set_model

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
    def __init__(self, dataset: str, model: str, start: int = 0, end: int = -1, mode: str = "atom"):
        # Initialize experiment runner
        logger.info(f"Initializing ExperimentRunner: dataset={dataset}, model={model}, start={start}, end={end}, mode={mode}")
        self.dataset = dataset
        self.start = start
        self.end = None if end == -1 else end
        self.interval = "full" if self.end is None else f"{start}-{end}"
        self.timestamp = time.time()
        self.mode = mode
        # Validate dataset support
        if dataset not in DATASET_CONFIGS:
            logger.error(f"Unsupported dataset: {dataset}") # Log error
            raise ValueError(f"Unsupported dataset: {dataset}")

        self.config = DATASET_CONFIGS[dataset]
        logger.info(f"Setting LLM model to: {model}") # Log model setting
        set_model(model)
        self.processor = AtomProcessor() # Instantiate processor

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
        
        # Pass different parameters based on scoring function
        if self.config.scoring_function == "score_math":
            entry["score"] = scoring_function(entry["answer"], groundtruth, self.dataset)
        else:
            entry["score"] = scoring_function(entry["answer"], groundtruth)
        return entry
    
    def update_score_log(self, accuracy: float) -> None:
        # Update score log
        log_entry = {
            "start": self.start,
            "end": self.end,
            "token": {"prompt": get_token()[0], "completion": get_token()[1]},
            "call_count": get_call_count(),
            "accuracy": accuracy,
        }
        
        score_log_file = LOG_DIR.format(dataset=self.dataset, size=self.interval) + "/score.json"
        existing_log = load_json(score_log_file) if os.path.exists(score_log_file) else {}
        count = get_file_count(LOG_DIR, self.interval, self.dataset, exclude_score=True)

        if self.dataset not in existing_log:
            existing_log[self.dataset] = {}
        existing_log[self.dataset][str(count)] = log_entry
        save_json(score_log_file, existing_log)
    
    async def run(self) -> float:
        # Run experiment and return accuracy
        logger.info(f"Starting experiment run: mode={self.mode}, dataset={self.dataset}, interval={self.interval}")
        print(f"Running {self.mode} experiment on {self.dataset} dataset from index {self.start} to {self.end}") # Keep print for user visibility

        # Load test set
        logger.info(f"Loading test data for {self.dataset} [{self.start}:{self.end}]")
        testset = load_data(self.dataset, "test")[self.start:self.end]
        logger.info(f"Loaded {len(testset)} items.") # Log item count
        logger.info("Gathering results...") # Log gathering start
        results = await self.gather_results(testset)
        logger.info(f"Gathered {len(results)} results.") # Log gathering end

        # Build results
        logger.info("Constructing result entries...") # Log construction start
        json_obj = []
        for i, (result, data) in enumerate(zip(results, testset)):
             try:
                 entry = self.construct_entry(result, data)
                 json_obj.append(entry)
             except Exception as e:
                 logger.error(f"Error constructing entry for item {self.start + i}: {e}", exc_info=True) # Log construction error
                 # Optionally add a placeholder or skip the entry
                 json_obj.append({
                     "problem": "Error processing item",
                     "groundtruth": data.get(self.config.answer_key, "N/A"),
                     "response": None,
                     "answer": None,
                     "log": result[1] if isinstance(result, tuple) and len(result) > 1 else None, # Log from processor if available
                     "score": 0,
                     "error": str(e)
                 })

        if not json_obj:
             logger.error("No results were successfully processed.") # Log if no results
             # Handle case with no results (e.g., return 0 accuracy or raise error)
             accuracy = 0.0
        else:
             accuracy = sum(entry["score"] for entry in json_obj) / len(json_obj)
        logger.info(f"Calculated accuracy: {accuracy:.4f}") # Log accuracy

        # Save results
        log_file = get_next_log_file(LOG_DIR, self.interval, self.dataset)
        logger.info(f"Saving detailed results to: {log_file}") # Log save path
        save_json(log_file, json_obj)

        # Update score log
        logger.info("Updating score log.") # Log score update
        self.update_score_log(accuracy)

        # Print result summary (Keep for user visibility)
        print(f"Unsolved: {round((1-accuracy) * len(json_obj))}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Time taken: {duration_formatter(time.time() - self.timestamp)}")
        logger.info("Experiment run finished.") # Log run finish

        return accuracy


async def optimize_dataset(dataset: str, model: str, start: int = 0, end: int = -1):
    # Optimize dataset questions and save to new file
    logger.info(f"Starting dataset optimization: dataset={dataset}, model={model}, start={start}, end={end}")
    print(f"Optimizing {dataset} dataset questions from index {start} to {end}") # Keep print
    timestamp = time.time()
    
    # Set model and instantiate processor
    logger.info(f"Setting LLM model to: {model}")
    set_model(model)
    processor = AtomProcessor() # Instantiate processor here
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
    
    args = parser.parse_args()

    # --- Setup Logging ---
    interval = "full" if args.end == -1 else f"{args.start}-{args.end}"
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
            mode=args.mode
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
