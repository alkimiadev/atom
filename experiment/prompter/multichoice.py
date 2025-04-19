import logging # Add logging

def direct(question: str):
	instruction = """
		You are a precise multiple choice question solver. Select the most correct option for the given question:

		QUESTION: {question}
		
		Please extend your chain of thought as much as possible; the longer the chain of thought, the better.
		
		You can freely reason in your response, but please enclose your final option within <answer>single letter of your chosen option</answer> tags.
	"""
	prompt = instruction.format(
		question=question,
	)
	return prompt

def multistep(question: str):
	instruction = """
		You are a precise multiple choice question solver. Break down complex questions into simpler sub-questions to select the most correct option:

		QUESTION: {question}
		
		Please extend your chain of thought as much as possible; the longer the chain of thought, the better.
		
		You can freely reason in your response, but please
		- Continuously raise sub-questions until the problem can be solved.
		- enclose your final option within <answer>single letter of your chosen option</answer> tags.
	"""
	prompt = instruction.format(
		question=question,
	)
	return prompt

def label(question: str, trajectory: str, answer: str):
	instruction = """
		You are tasked with breaking down a multiple choice question reasoning process into sub-questions.

		Original Question: {question}
		Complete Reasoning Process: {trajectory}

		Instructions:
		1. Break down the reasoning process into a series of sub-questions
		2. Each sub-question should:
		   - Be written in interrogative form
		   - Have a clear answer
		   - List its other sub-questions' indexes it depends (0-based, can be an empty list)
		3. Dependencies are defined as information needed to answer the current sub-question that:
		   - Does NOT come directly from the original question
		   - MUST come from the answers of previous sub-questions
	"""
	formatter = f"""
		Format your response using the following XML structure:
		<response>
		  <thought>The thought process of how to step by step propose the sub-questions until the answer of the original question in the given reasoning process is obtained</thought>
		  <sub-questions>
		    <sub-question>
		      <description>The description of the sub-question</description>
		      <answer>The answer to the sub-question</answer>
		      <depend>
		        <index>Index of prerequisite sub-question (0-based)</index>
		        <!-- Add more <index> tags if needed -->
		      </depend>
		      <!-- If no dependencies, leave <depend> empty or omit it -->
		    </sub-question>
		    <!-- Add more <sub-question> blocks as needed -->
		  </sub-questions>
		  <answer>{answer}</answer> <!-- The final single-letter answer to the original question -->
		</response>
	"""
	return (instruction + formatter).format(question=question, trajectory=trajectory, answer=answer)

def contract(question: str, decompose_result: dict, independent: list, dependent: list):
	instruction = """
		You are a multiple choice question solver specializing in optimizing step-by-step reasoning processes. Your task is to optimize the existing reasoning trajectory into a more efficient, single self-contained question.
		
		For the original question: {question}
		
		Here are step-by-step reasoning process:
		{response}
		
		{sub_questions}
		
		Here are explanations of key concepts:
		1. self-contained: The optimized question must be solvable independently, without relying on any external information
		2. efficient: The optimized question must be simpler than the original, requiring fewer reasoning steps and having a clearer reasoning process (these steps are reduced because some solved sub-problems become known conditions in the optimized question or are excluded as incorrect explorations)
		
		Note: Since this is a multiple choice question, the optimized question must completely retain the options of the original question.
		
		You can freely reason in your response, but please enclose the your optimized question within <question></question> tags
	"""
	sub_questions = """
		The following sub-questions and their answers can serve as known conditions:
		{independent}

		The descriptions of the following questions can be used to form the description of the optimized problem:
		{dependent}
		
		"""
	answer = decompose_result["answer"]
	for sub_q in independent:
		sub_q.pop("depend", None)
	for sub_q in dependent:
		sub_q.pop("depend", None)
		
	sub_questions = sub_questions.format(independent=independent, dependent=dependent)
	return instruction.format(question=question, answer=answer, response=decompose_result["response"], sub_questions=sub_questions)

def ensemble(question: str, solutions: list):
	instruction = """
		You are a precise multiple choice question solver. Compare then synthesize the best answer from multiple solutions to select the most correct option:

		QUESTION: {question}

		SOLUTIONS:
		{solutions}
		
		Extend your chain of thought as much as possible; the longer the chain of thought, the better.

		You can freely reason in your response, even propose new reasoning to get a better answer than all solutions, but please mark the final option with <answer>single letter of your chosen option</answer> tags
	"""
	
	solutions_str = ""
	for i, solution in enumerate(solutions):
		solutions_str += f"solution {i}: {solution}\n"
	prompt = instruction.format(question=question, solutions=solutions_str)
	return prompt

	return prompt

logger = logging.getLogger(__name__) # Add logger

def check_answer(answer):
	if not isinstance(answer, str):
		return False
	if len(answer) == 1 and answer.isalpha():
		return True
	if len(answer) == 3 and answer.startswith('(') and answer.endswith(')'):
		return True
	return False

def check(name: str, result: dict, *args):
	# Basic check: Ensure result is a non-empty dictionary
	if not isinstance(result, dict):
		logger.debug(f"Check '{name}': Failed - result is not a dict (type: {type(result)})")
		return False
	if not result:
		logger.debug(f"Check '{name}': Failed - result dict is empty (likely XML extraction failure)")
		return False

	if name in ["cot", "direct", "multistep", "ensemble"]:
		# Expecting <answer> tag
		if 'answer' not in result:
			logger.debug(f"Check '{name}': Failed - 'answer' key missing. Keys: {result.keys()}")
			return False
		if not check_answer(result['answer']):
			logger.debug(f"Check '{name}': Failed - 'answer' content failed check_answer ('{result['answer']}')")
			return False
	elif name == "label":
		# Expecting <response><thought>...</thought><sub-questions>...</sub-questions><answer>...</answer></response>
		if not all(k in result for k in ['thought', 'sub-questions', 'answer']):
			logger.debug(f"Check '{name}': Failed - Missing 'thought', 'sub-questions', or 'answer'. Keys: {result.keys()}")
			return False
		if not check_answer(result['answer']):
			logger.debug(f"Check '{name}': Failed - Top-level 'answer' failed check_answer ('{result['answer']}')")
			return False

		sub_questions_data = result['sub-questions']
		sub_questions_list = []
		if isinstance(sub_questions_data, dict) and 'sub-question' in sub_questions_data:
			sub_questions_list = sub_questions_data['sub-question']
			if not isinstance(sub_questions_list, list): sub_questions_list = [sub_questions_list]
		elif isinstance(sub_questions_data, list): # Allow list directly?
			sub_questions_list = sub_questions_data
		else:
			logger.debug(f"Check '{name}': Failed - 'sub-questions' has unexpected structure. Data: {sub_questions_data}")
			return False

		for i, sub_q in enumerate(sub_questions_list):
			if not isinstance(sub_q, dict) or not all(k in sub_q for k in ['description', 'answer', 'depend']):
				logger.debug(f"Check '{name}': Failed - Sub-question {i} missing keys or not a dict. Keys: {sub_q.keys() if isinstance(sub_q, dict) else 'N/A'}")
				return False
			# Sub-question answer can be any string here, no specific format check needed like check_answer
			if not isinstance(sub_q['answer'], str):
				logger.debug(f"Check '{name}': Failed - Sub-question {i} 'answer' is not a string.")
				return False
			# Validate 'depend' structure (similar to math prompter)
			if 'depend' in sub_q and sub_q['depend'] is not None:
				depend_data = sub_q['depend']
				if isinstance(depend_data, dict) and 'index' in depend_data:
					indices = depend_data['index']
					if not isinstance(indices, list): indices = [indices]
					if not all(isinstance(idx, (str, int)) for idx in indices):
						logger.debug(f"Check '{name}': Failed - Sub-question {i} 'depend/index' contains non-str/int values.")
						return False
				elif not isinstance(depend_data, dict):
					logger.debug(f"Check '{name}': Failed - Sub-question {i} 'depend' is not a dict or None.")
					return False

	elif name == "contract":
		# Expecting <question> tag
		if 'question' not in result:
			logger.debug(f"Check '{name}': Failed - 'question' key missing. Keys: {result.keys()}")
			return False

	logger.debug(f"Check '{name}': Passed.")
	return True
