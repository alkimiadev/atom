import re
import os
import json
import re
import string
import logging # Add logging import
from collections import Counter
from typing import List, Union

logger = logging.getLogger(__name__) # Add logger for this module

def extract_json(string):
	logger.debug(f"Attempting to extract JSON from string (length {len(string)}): {string[:200]}...") # Log input (truncated)
	try:
		string = string.strip()
		start, end = string.find("{"), string.rfind("}")
		if start != -1 and end != -1:
			string = string[start : end + 1]
		# Attempt to remove common control characters before parsing
		# Allow tabs (\t), newlines (\n), carriage returns (\r)
		cleaned_string = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', string)
		# --- Attempt basic JSON auto-fixing ---
		fixed_string = cleaned_string
		# Heuristic: Add comma between } and " or } and { or ] and { or ] and "
		fixed_string = re.sub(r'}(?=\s*")', '},', fixed_string)
		fixed_string = re.sub(r'}(?=\s*{)', '},', fixed_string)
		fixed_string = re.sub(r'](?=\s*{)', '],', fixed_string)
		fixed_string = re.sub(r'](?=\s*")', '],', fixed_string)
		# --- End auto-fixing attempt ---
		try:
			# Try parsing the potentially fixed string first
			logger.debug(f"Attempting JSON load on potentially fixed string (first 200 chars): {fixed_string[:200]}...")
			json_data = json.loads(fixed_string)
			logger.debug(f"Successfully extracted JSON from potentially fixed string.") # Log success
			return json_data
		except json.JSONDecodeError as e_fix:
			logger.warning(f"JSON parsing failed after attempting fixes: {e_fix}. Trying original cleaned string.")
			# Fallback to original cleaned string if fix fails or introduces new errors
			try:
				json_data = json.loads(cleaned_string)
				logger.debug(f"Successfully extracted JSON from original cleaned string.") # Log success
				return json_data
			except json.JSONDecodeError as e_orig:
				# Log specific JSON decode errors, including position if available
				logger.error(f"Failed to decode JSON even on original cleaned string: {e_orig} at char {e_orig.pos}", exc_info=False) # exc_info=False to avoid duplicate traceback
				# Log original raw string, cleaned string, AND fixed string on final error for comparison
				logger.debug(f"Original RAW string causing JSON decode error: {string}")
				logger.debug(f"Cleaned string causing JSON decode error: {cleaned_string}")
				logger.debug(f"Auto-fixed string causing JSON decode error: {fixed_string}")
				return None # Return None on JSON failure
	except Exception as e:
		# Catch other potential exceptions during extraction
		logger.error(f"Failed during JSON extraction (not JSONDecodeError): {e}", exc_info=True)
		logger.debug(f"Original RAW string causing other extraction error: {string}") # Log original raw string on error
		return None # Return None on other failures

def extract_xml(string):
	logger.debug(f"Attempting to extract XML from string (length {len(string)}): {string[:200]}...") # Log input (truncated)
	try:
		# Remove any leading/trailing whitespace
		string = string.strip()

		# Use regex to find all tag-content pairs
		pattern = r"<([\w-]+)>(.*?)</\1>"
		matches = re.finditer(pattern, string)

		result = {}

		# Process each match, later matches will overwrite earlier ones
		for match in matches:
			tag = match.group(1)
			content = match.group(2).strip()

			# Try to convert content to number if possible
			try:
				if content.isdigit():
					value = int(content)
				else:
					value = float(content)
			except:
				value = content

			# Simply update the value, overwriting any previous value
			result[tag] = value

		logger.debug(f"Successfully extracted XML: {result}") # Log success
		return result
	except Exception as e:
		logger.error(f"Failed to extract XML: {e}", exc_info=True) # Log error
		logger.debug(f"Original string causing XML error: {string}") # Log original string on error
		return {}

def check_json(json_obj, keys: list):
	if not isinstance(json_obj, dict):
		return False
	for key in keys:
		if key not in json_obj.keys():
			return False
	return True

def save_json(filepath, data):
	with open(filepath, "w") as f:
		json.dump(data, f, indent=2)

def load_json(filepath):
	if os.path.exists(filepath):
		with open(filepath, "r") as f:
			return json.load(f)
	return {}

def duration_formatter(seconds):
	hours = seconds // 3600
	minutes = (seconds % 3600) // 60
	seconds = seconds % 60
	if hours > 0:
		return f"{int(hours):02d}h:{int(minutes):02d}m:{int(seconds):02d}s"
	elif minutes > 0:
		return f"{int(minutes):02d}m:{int(seconds):02d}s"
	else:
		return f"{int(seconds):02d}s"

def calculate_depth(sub_questions: list):
	logger.debug(f"Calculating depth for {len(sub_questions)} sub-questions. Input (first 500 chars): {str(sub_questions)[:500]}") # Log input
	try:
		n = len(sub_questions)

		# Initialize distances matrix with infinity
		distances = [[float("inf")] * n for _ in range(n)]

		# Set direct dependencies
		for i, sub_q in enumerate(sub_questions):
			logger.debug(f"calculate_depth: Processing sub_q #{i}: Type={type(sub_q)}, Value={str(sub_q)[:200]}") # Log sub_q
			# Distance to self is 0
			distances[i][i] = 0
			# Set direct dependencies with distance 1
			# Check if sub_q is a dictionary before calling .get()
			if isinstance(sub_q, dict):
				dependencies = sub_q.get("depend", [])
				logger.debug(f"calculate_depth: Sub_q #{i} dependencies: {dependencies}") # Log dependencies
				for dep in dependencies:
					logger.debug(f"calculate_depth: Processing dependency 'dep': Type={type(dep)}, Value={dep}") # Log dep
					# Check if dep is an integer before comparison
					if isinstance(dep, int):
						# Add check for dependency index bounds
						if dep >= n:
							logger.error(f"Invalid dependency index in calculate_depth: sub_question index i={i}, dependency index dep={dep}, but n={n}. Skipping this dependency.")
							continue # Skip this invalid dependency
						distances[dep][i] = 1
					else:
						logger.warning(f"calculate_depth: Dependency 'dep' for sub_q #{i} is not an integer (Type: {type(dep)}, Value: {dep}). Skipping.")
			else:
				logger.warning(f"calculate_depth: Sub_q #{i} is not a dictionary. Skipping dependency processing for this item.")


		# Floyd-Warshall algorithm to find shortest paths
		for k in range(n):
			for i in range(n):
				for j in range(n):
					if distances[i][k] != float("inf") and distances[k][j] != float("inf"):
						distances[i][j] = min(
							distances[i][j], distances[i][k] + distances[k][j]
						)

		# Find maximum finite distance
		max_depth = 0
		for i in range(n):
			for j in range(n):
				if distances[i][j] != float("inf"):
					max_depth = max(max_depth, distances[i][j])

		logger.debug(f"Calculated max depth: {max_depth}")
		return int(max_depth)
	except Exception as e:
		logger.error(f"Failed to calculate depth: {e}", exc_info=True)
		logger.warning("Returning default depth of 3 due to calculation error.")
		return 3
def get_next_log_file(log_dir, size, dataset):
	directory = log_dir.format(dataset=dataset, size=size)
	os.makedirs(directory, exist_ok=True)
	
	# 只计算数字命名的json文件，排除score.json
	files = [f for f in os.listdir(directory) if f.endswith('.json') and f != 'score.json']
	
	# 找出最大的数字编号
	max_num = 0
	for f in files:
		try:
			num = int(f.split('.')[0])
			max_num = max(max_num, num)
		except ValueError:
			continue
	
	return os.path.join(directory, f"{max_num + 1}.json")

def get_file_count(log_dir, interval, dataset, exclude_score=False):
	directory = log_dir.format(dataset=dataset, size=interval)
	if not os.path.exists(directory):
		os.makedirs(directory, exist_ok=True)
		return 0
	
	files = os.listdir(directory)
	if exclude_score:
		# 排除score.json，只计算数字命名的json文件
		files = [f for f in files if f != "score.json"]
	
	return len(files)

## hotpotqa
def normalize_answer(s):

	def remove_articles(text):
		return re.sub(r"\b(a|an|the)\b", " ", text)

	def white_space_fix(text):
		return " ".join(text.split())

	def remove_punc(text):
		exclude = set(string.punctuation)
		return "".join(ch for ch in text if ch not in exclude)

	def lower(text):
		return text.lower()

	return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
	return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction, ground_truth):
	normalized_prediction = normalize_answer(prediction)
	normalized_ground_truth = normalize_answer(ground_truth)

	ZERO_METRIC = (0, 0, 0)

	if (
		normalized_prediction in ["yes", "no", "noanswer"]
		and normalized_prediction != normalized_ground_truth
	):
		return ZERO_METRIC
	if (
		normalized_ground_truth in ["yes", "no", "noanswer"]
		and normalized_prediction != normalized_ground_truth
	):
		return ZERO_METRIC

	prediction_tokens = normalized_prediction.split()
	ground_truth_tokens = normalized_ground_truth.split()
	common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
	num_same = sum(common.values())
	if num_same == 0:
		return ZERO_METRIC
	precision = 1.0 * num_same / len(prediction_tokens)
	recall = 1.0 * num_same / len(ground_truth_tokens)
	f1 = (2 * precision * recall) / (precision + recall)
	return f1, precision, recall


def score_mh(prediction: str, groundtruth: Union[str, list]):
	logger.debug(f"Scoring MH: Prediction='{prediction}', Groundtruth='{groundtruth}'")
	try:
		if isinstance(groundtruth, list):
			f1 = max([f1_score(prediction, gt)[0] for gt in groundtruth])
		else:
			f1 = f1_score(prediction, groundtruth)[0]
		logger.debug(f"MH Score (F1): {f1}")
		return f1
	except Exception as e:
		logger.error(f"Failed to score MH: Prediction='{prediction}', Groundtruth='{groundtruth}', Error: {e}", exc_info=True)
		return 0

# math
def extract_boxed(s):
	import re

	pattern = r"\\boxed{((?:[^{}]|{(?:[^{}]|{[^{}]*})*?)*)}"
	match = re.search(pattern, s)
	if match:
		return match.group(1)
	return ""


def eval_math(s):
	try:
		return eval(str(s).replace(",", ""))
	except:
		return 0


def score_math(prediction, groundtruth, dataset="aime"):
	logger.debug(f"Scoring Math ({dataset}): Prediction='{prediction}', Groundtruth='{groundtruth}'")
	try:
		score = 0 # Default score
		pred_eval = eval_math(prediction)
		if dataset == "math":
			gt_eval = eval_math(extract_boxed(groundtruth))
			score = 1 if pred_eval == gt_eval else 0
		elif dataset == "gsm8k":
			gt_eval = eval_math(groundtruth.split("####")[1])
			score = 1 if pred_eval == gt_eval else 0
		elif dataset == "aime":
			gt_eval = eval_math(groundtruth)
			score = 1 if pred_eval == gt_eval else 0
		else:
			 logger.warning(f"Unknown dataset type for math scoring: {dataset}")

		logger.debug(f"Math Score: {score}")
		return score
	except Exception as e:
		logger.error(f"Failed to score Math ({dataset}): Prediction='{prediction}', Groundtruth='{groundtruth}', Error: {e}", exc_info=True)
		return 0


# logic
def score_mc(prediction, target):
	logger.debug(f"Scoring MC: Prediction='{prediction}', Target='{target}'")
	if not prediction or not target:
		logger.debug("MC Score: 0 (Missing prediction or target)")
		return 0

	prediction = str(prediction).upper()
	target = str(target).upper()

	def normalize_answer(answer):
		# Remove any brackets and convert to uppercase
		return answer.replace("(", "").replace(")", "").upper()

	score = 1 if normalize_answer(prediction) == normalize_answer(target) else 0
	logger.debug(f"MC Score: {score}")
	return score
