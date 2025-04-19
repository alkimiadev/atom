import json
import logging # Add logging

def cot(question: str, contexts: str = None):
	instruction = """
		Please solve the multi-hop question below based on the following contexts step by step:

		QUESTION: 
		{question}

		CONTEXTS: 
		{contexts}
	"""
	formatter = """

		Provide your response using the following XML structure:
		<response>
		  <thought>Give your step-by-step reasoning process</thought>
		  <answer>Your precise answer</answer>
		</response>
	"""
	prompt = (instruction + formatter).format(question=question, contexts=contexts)
	return prompt

def direct(question: str, contexts: str = None):
	instruction = """
		You are a precise question-answering solver. Answer the following question using only the provided contexts:

		QUESTION: 
		{question}

		CONTEXTS: 
		{contexts}

		INSTRUCTIONS:
		1. Answer Selection Rules:
		   a) Use ONLY information from the given contexts
		   b) For yes/no questions: Answer with exactly "yes" or "no"
		   c) For other questions: Extract a precise answer that is:
			  - CONTINUOUS: Must be an unbroken segment from the text
			  - EXACT: Use the original text without modifications
			  - MINIMAL: Include only the essential information

		2. Supporting Evidence:
		   - Select ALL relevant sentences that lead to your answer
		   - Include complete context when needed
		   - You may use ellipsis (...) to connect relevant parts of long sentences
		   
		   EXAMPLE:
		   Question: "Where was the rock band Letters to Cleo formed?"
		   Supporting Sentences: 
		   ✓ Good: "Letters to Cleo are an alternative rock band from Boston, Massachusetts..."
		   × Bad: "The band was formed in Boston, Massachusetts" (lacks subject reference)

		3. Answer Extraction Guidelines:
		   a) CONTINUOUS text only:
			  Question: "Where is BTS from?"
			  Context: "BTS is a South Korean boy band formed in Seoul"
			  ✓ CORRECT: "Seoul"
			  × WRONG: "Seoul, South Korea" (combining segments)

		   b) EXACT text:
			  Question: "When was Nixon president?"
			  Context: "Nixon was president from 1969 until 1974"
			  ✓ CORRECT: "1969 until 1974"
			  × WRONG: "1969-1974" (modified text)

		   c) MINIMAL answer:
			  Question: "What was Tesla's profession?"
			  Context: "Nikola Tesla was a brilliant Serbian-American inventor"
			  ✓ CORRECT: "inventor"
			  × WRONG: "brilliant Serbian-American inventor" (includes unnecessary details)

		4. Important:
		   - Handle unclear questions by focusing on the main intent
		   - Avoid common pitfalls like combining disconnected information
		   - Prioritize precision over completeness
		
		5. Robustness:
			Sometimes the question may have some errors, leading to a situation where there is actually no answer in the context. I hope you can infer what the questioner is actually asking and then respond according to the above process.
	"""
	
	formatter = """
	Provide your response using the following XML structure:
	<response>
	  <question>{question}</question>
	  <thought>Give your step-by-step thought process here</thought>
	  <supporting_sentences>
	    <sentence>Include ALL sentences needed to justify your answer</sentence>
	    <sentence>Use ... for long sentences when appropriate</sentence>
	    <!-- Add more <sentence> tags as needed -->
	  </supporting_sentences>
	  <answer>Your precise answer following the instructions above, or "none" if no answer can be found</answer>
	</response>
	"""
	prompt = (instruction + formatter).format(question=json.dumps(question), contexts=contexts) # Keep question JSON escaped for XML safety
	return prompt

def multistep(question: str, contexts: str = None):
	instruction = """
		You are a precise question-answering solver. Breaks down multi-hop questions into single-hop sub-questions to answer the following question using only the provided contexts:

		QUESTION: 
		{question}

		CONTEXTS: 
		{contexts}

		INSTRUCTIONS:
		1. Answer Selection Rules:
		   a) Use ONLY information from the given contexts
		   b) For yes/no questions: Answer with exactly "yes" or "no"
		   c) For other questions: Extract a precise answer that is:
			  - CONTINUOUS: Must be an unbroken segment from the text
			  - EXACT: Use the original text without modifications
			  - MINIMAL: Include only the essential information

		2. Supporting Evidence:
		   - Select ALL relevant sentences that lead to your answer
		   - Include complete context when needed
		   - You may use ellipsis (...) to connect relevant parts of long sentences
		   
		   EXAMPLE:
		   Question: "Where was the rock band Letters to Cleo formed?"
		   Supporting Sentences: 
		   ✓ Good: "Letters to Cleo are an alternative rock band from Boston, Massachusetts..."
		   × Bad: "The band was formed in Boston, Massachusetts" (lacks subject reference)

		3. Answer Extraction Guidelines:
		   a) CONTINUOUS text only:
			  Question: "Where is BTS from?"
			  Context: "BTS is a South Korean boy band formed in Seoul"
			  ✓ CORRECT: "Seoul"
			  × WRONG: "Seoul, South Korea" (combining segments)

		   b) EXACT text:
			  Question: "When was Nixon president?"
			  Context: "Nixon was president from 1969 until 1974"
			  ✓ CORRECT: "1969 until 1974"
			  × WRONG: "1969-1974" (modified text)

		   c) MINIMAL answer:
			  Question: "What was Tesla's profession?"
			  Context: "Nikola Tesla was a brilliant Serbian-American inventor"
			  ✓ CORRECT: "inventor"
			  × WRONG: "brilliant Serbian-American inventor" (includes unnecessary details)

		4. Important:
		   - Handle unclear questions by focusing on the main intent
		   - Avoid common pitfalls like combining disconnected information
		   - Prioritize precision over completeness
		   
		5. Robustness:
			Sometimes the question may have some errors, leading to a situation where there is actually no answer in the context. I hope you can infer what the questioner is actually asking and then respond according to the above process.
	"""
	
	formatter = """
	Provide your response using the following XML structure:
	<response>
	  <question>{question}</question>
	  <thought>Give your step-by-step thought process here</thought>
	  <sub-questions>
	    <sub-question>
	      <description>The description of the sub-question</description>
	      <supporting_sentences>
	        <sentence>Include ALL sentences needed to justify your answer to this sub-question</sentence>
	        <sentence>Use ... for long sentences when appropriate</sentence>
	        <!-- Add more <sentence> tags as needed -->
	      </supporting_sentences>
	      <answer>Answer to this sub-question</answer>
	    </sub-question>
	    <!-- Add more <sub-question> blocks as needed -->
	  </sub-questions>
	  <conclusion>Explain how the sub-answers combine to answer the main question</conclusion>
	  <answer>Your precise answer to the main question, or "none" if no answer can be found</answer>
	</response>
	"""
	prompt = (instruction + formatter).format(question=json.dumps(question), contexts=contexts) # Keep question JSON escaped for XML safety
	return prompt

def label(question: str, result: dict):
	instruction = f"""
		For the original question: {question},
		We have broken it down into the following sub-questions:
		SUB-QUESTIONS:
		{result["sub-questions"]}
		And obtained a complete reasoning process for the original question:
		{result}
		We define the dependency relationship between sub-questions as: which information in the current sub-question description does not come directly from the original question and contexts, but from the results of other sub-questions.
		
		You are a question answering expert specializing in analyzing the dependency relationships between these sub-questions. Please return an XML structure that expresses a complete reasoning trajectory for the original question, including the question, answer, supporting evidence, and dependency relationships of each sub-question. The dependency relationships are represented by the indices of the dependent sub-questions in SUB-QUESTIONS, starting from zero.
	"""
	
	formatter = '''
		Format your response using the following XML structure:
		<response>
		  <thought>Give your thought process here</thought>
		  <sub-questions>
''' # Start XML structure
	formatted_sub_qs = []
	if isinstance(result.get("sub-questions"), list): # Ensure sub-questions is a list
		for sub_q_item in result["sub-questions"]:
			sub_q_dict = None
			# Check if it's already a dictionary
			if isinstance(sub_q_item, dict):
				sub_q_dict = sub_q_item
			# If it's a string, try to parse it as JSON
			elif isinstance(sub_q_item, str):
				try:
					sub_q_dict = json.loads(sub_q_item)
					# Ensure the parsed result is actually a dictionary
					if not isinstance(sub_q_dict, dict):
						print(f"Warning: Parsed sub-question string but did not get a dict: {sub_q_item}")
						sub_q_dict = None # Reset if parsing didn't yield a dict
				except json.JSONDecodeError:
					print(f"Warning: Failed to parse sub-question string as JSON: {sub_q_item}")
					sub_q_dict = None # Parsing failed
   
			# Process if we successfully obtained a dictionary
			if sub_q_dict:
				# Safely get values, providing defaults if keys are missing
				desc = sub_q_dict.get("description", "N/A")
				ans = sub_q_dict.get("answer", "N/A")
				sup_sent = sub_q_dict.get("supporting_sentences", [])
				# Format supporting sentences as XML elements
				sup_sent_xml = ""
				if isinstance(sup_sent, list):
					for sent in sup_sent:
						# Basic XML escaping for content
						escaped_sent = str(sent).replace('&', '&').replace('<', '<').replace('>', '>')
						sup_sent_xml += f'          <sentence>{escaped_sent}</sentence>\n'
				else: # Handle case where supporting_sentences might not be a list as expected
					sup_sent_xml = f'          <sentence>Error: Expected list, got {type(sup_sent)}</sentence>\n'
			
				formatted_sub_qs.append(
					f'''		    <sub-question>
				    <description>{str(desc).replace('&', '&').replace('<', '<').replace('>', '>')}</description>
				    <answer>{str(ans).replace('&', '&').replace('<', '<').replace('>', '>')}</answer>
				    <supporting_sentences>
{sup_sent_xml}		      </supporting_sentences>
				    <depend>
				      <index>Index of prerequisite sub-question (0-based)</index>
				      <!-- Add more <index> tags if needed, or leave empty -->
				    </depend>
				  </sub-question>'''
				)
			else:
				# Log a warning if the item was neither a dict nor a valid JSON string dict
				if not isinstance(sub_q_item, dict) and not isinstance(sub_q_item, str):
					print(f"Warning: Skipping invalid sub-question item (type {type(sub_q_item)}): {sub_q_item}")
				# The warnings for failed parsing or wrong parsed type are handled above
				pass
   
	# Join the valid formatted sub-questions with newlines
	formatter += "\n".join(formatted_sub_qs)
	# Close XML structure
	formatter += "\n		  </sub-questions>\n		</response>"
	  
	# Need to import json at the top of the file for json.dumps
	return instruction + formatter

def contract(question: str, decompose_result: dict, independent: list, dependent: list, contexts: str = None):
	instruction = """
		You are a precise question-answering solver specializing in optimizing step-by-step reasoning processes. Your task is to optimize the existing reasoning trajectory into a more efficient, single-hop and self-contained question.
		
		For the original question: {question}
		
		Here are the contexts that can be used to answer the original question (but only some of them can be directly used to solve the question):
		{contexts}
		
		Here are step-by-step reasoning process:
		{response}
		
		{sub_questions}
		
		Here are explanations of key concepts:
		1. self-contained: The optimized question must be solvable independently, without relying on any external information
		2. efficient: The optimized question must be simpler than the original, requiring fewer reasoning steps and having a clearer reasoning process (these steps are reduced because some solved sub-problems become known conditions in the optimized question or are excluded as incorrect explorations)
		
		You can freely reason in your response, but please enclose the your optimized question within <question></question> tags. Enclose the complete context needed to answer the optimized question within <context></context> tags. IMPORTANT: The content inside the <context> tags MUST be the plain text of the relevant context sentences, NOT a JSON or dictionary structure.
	"""
	sub_questions = """
		The following sub-questions and their answers can serve as known conditions:
		{independent}

		The descriptions of the following questions can be used to form the description of the optimized problem:
		{dependent}
		"""
	for sub_q in independent:
		sub_q.pop("depend", None)
	for sub_q in dependent:
		sub_q.pop("depend", None)
		
	sub_questions = sub_questions.format(independent=independent, dependent=dependent)
	return instruction.format(question=question, contexts=contexts, response=decompose_result, sub_questions=sub_questions)

def ensemble(question: str, solutions: list, contexts: str = None):
	instruction = """
		You are a precise question answering expert. Compare then synthesize the best answer from multiple solutions to solve the following question.
		
		QUESTION:
		{question}

		CONTEXTS:
		{contexts}

		SOLUTIONS:
		{solutions}

		INSTRUCTIONS:
		1. Answer Selection Rules:
		   a) Use ONLY information from the given contexts
		   b) For yes/no questions: Answer with exactly "yes" or "no"
		   c) For other questions: Extract a precise answer that is:
			  - CONTINUOUS: Must be an unbroken segment from the text
			  - EXACT: Use the original text without modifications
			  - MINIMAL: Include only the essential information

		2. Supporting Evidence:
		   - Select ALL relevant sentences that lead to your answer
		   - Include complete context when needed
		   - You may use ellipsis (...) to connect relevant parts of long sentences
		   
		   EXAMPLE:
		   Question: "Where was the rock band Letters to Cleo formed?"
		   Supporting Sentences: 
		   ✓ Good: "Letters to Cleo are an alternative rock band from Boston, Massachusetts..."
		   × Bad: "The band was formed in Boston, Massachusetts" (lacks subject reference)

		3. Answer Extraction Guidelines:
		   a) CONTINUOUS text only:
			  Question: "Where is BTS from?"
			  Context: "BTS is a South Korean boy band formed in Seoul"
			  ✓ CORRECT: "Seoul"
			  × WRONG: "Seoul, South Korea" (combining segments)

		   b) EXACT text:
			  Question: "When was Nixon president?"
			  Context: "Nixon was president from 1969 until 1974"
			  ✓ CORRECT: "1969 until 1974"
			  × WRONG: "1969-1974" (modified text)

		   c) MINIMAL answer:
			  Question: "What was Tesla's profession?"
			  Context: "Nikola Tesla was a brilliant Serbian-American inventor"
			  ✓ CORRECT: "inventor"
			  × WRONG: "brilliant Serbian-American inventor" (includes unnecessary details)

		4. Important:
		   - Handle unclear questions by focusing on the main intent
		   - Avoid common pitfalls like combining disconnected information
		   - Prioritize precision over completeness
		
		5. Robustness:
			Sometimes the question may have some errors, leading to a situation where there is actually no answer in the context. I hope you can infer what the questioner is actually asking and then respond according to the above process.
	"""
	
	formatter = """
		Format your response using the following XML structure:
		<response>
		  <question>{question}</question>
		  <thought>Explain your analysis of the different results and why you chose the final answer</thought>
		  <supporting_sentences>
		    <sentence>Include ALL sentences needed to justify your answer</sentence>
		    <sentence>Use ... for long sentences when appropriate</sentence>
		    <!-- Add more <sentence> tags as needed -->
		  </supporting_sentences>
		  <answer>The most reliable answer following the answer instructions</answer>
		</response>
	"""
	solutions_str = ""
	for i, solution in enumerate(solutions):
		solutions_str += f"solution {i}: {solution}\n"
	prompt = (instruction + formatter).format(question=json.dumps(question), contexts=contexts, solutions=solutions_str) # Keep question JSON escaped for XML safety
	return prompt

	return prompt

logger = logging.getLogger(__name__) # Add logger

# utilization
def contexts(obj: dict, dataset: str):
	if dataset == "hotpotqa":
		context = []
		for i in range(len(obj["context"]["sentences"])):
			context.append(" ".join(obj["context"]["sentences"][i]))
		return context
		# return obj["context"]
	elif dataset == "longbench":
		return obj["context"]
	else:
		raise ValueError("Unknown dataset format: neither 'context' nor 'paragraphs' field found")

def check(name: str, result: dict, *args):
	# Basic check: Ensure result is a non-empty dictionary
	if not isinstance(result, dict):
		logger.debug(f"Check '{name}': Failed - result is not a dict (type: {type(result)})")
		return False
	if not result:
		logger.debug(f"Check '{name}': Failed - result dict is empty (likely XML extraction failure)")
		return False

	# Helper to check for non-empty string answer
	def is_valid_answer(answer):
		return isinstance(answer, str) and answer.lower() not in ["null", "none", ""]

	# Helper to check supporting sentences structure
	def check_supporting_sentences(sentences_data, log_prefix):
		if not isinstance(sentences_data, dict) or 'sentence' not in sentences_data:
			logger.debug(f"{log_prefix}: Failed - 'supporting_sentences' structure invalid or missing 'sentence' key. Data: {sentences_data}")
			return False
		sentences = sentences_data['sentence']
		if not isinstance(sentences, list): sentences = [sentences] # Handle single sentence
		if not all(isinstance(s, str) for s in sentences):
			logger.debug(f"{log_prefix}: Failed - Not all items in 'supporting_sentences/sentence' are strings.")
			return False
		return True

	if name == "cot":
		# Expecting <response><thought>...</thought><answer>...</answer></response>
		if not all(k in result for k in ['thought', 'answer']):
			logger.debug(f"Check '{name}': Failed - Missing 'thought' or 'answer'. Keys: {result.keys()}")
			return False
		if not is_valid_answer(result['answer']):
			logger.debug(f"Check '{name}': Failed - Invalid answer ('{result['answer']}')")
			return False

	elif name == "direct":
		logger.debug(f"Check '{name}': Validating result dict: {result}") # Log the full result dict
		# Expecting <response><question>...</question><thought>...</thought><supporting_sentences>...</supporting_sentences><answer>...</answer></response>
		required_keys = ['question', 'thought', 'supporting_sentences', 'answer']
		if not all(k in result for k in required_keys):
			logger.debug(f"Check '{name}': Failed - Missing keys. Expected: {required_keys}, Got: {result.keys()}")
			return False
		# Log supporting sentences before checking
		supporting_sentences_data = result.get('supporting_sentences')
		logger.debug(f"Check '{name}': Validating supporting_sentences data (type {type(supporting_sentences_data)}): {supporting_sentences_data}")
		if not check_supporting_sentences(supporting_sentences_data, f"Check '{name}'"):
			# Specific log inside check_supporting_sentences will indicate the failure reason
			return False
		# Log answer before checking
		answer_data = result.get('answer')
		logger.debug(f"Check '{name}': Validating answer data (type {type(answer_data)}): '{answer_data}'")
		if not is_valid_answer(answer_data):
			logger.debug(f"Check '{name}': Failed - Invalid answer ('{answer_data}')")
			return False

	elif name == "multistep":
		# Expecting <response><question>...</question><thought>...</thought><sub-questions>...</sub-questions><conclusion>...</conclusion><answer>...</answer></response>
		required_keys = ['question', 'thought', 'sub-questions', 'conclusion', 'answer']
		if not all(k in result for k in required_keys):
			logger.debug(f"Check '{name}': Failed - Missing keys. Expected: {required_keys}, Got: {result.keys()}")
			return False
		if not is_valid_answer(result['answer']):
			logger.debug(f"Check '{name}': Failed - Invalid answer ('{result['answer']}')")
			return False
		# Check sub-questions structure
		sub_questions_data = result['sub-questions']
		if not isinstance(sub_questions_data, dict) or 'sub-question' not in sub_questions_data:
			logger.debug(f"Check '{name}': Failed - 'sub-questions' structure invalid or missing 'sub-question' key. Data: {sub_questions_data}")
			return False
		sub_questions_list = sub_questions_data['sub-question']
		if not isinstance(sub_questions_list, list): sub_questions_list = [sub_questions_list] # Handle single
		for i, sub_q in enumerate(sub_questions_list):
			sub_q_keys = ['description', 'supporting_sentences', 'answer']
			if not isinstance(sub_q, dict) or not all(k in sub_q for k in sub_q_keys):
				logger.debug(f"Check '{name}': Failed - Sub-question {i} missing keys or not a dict. Expected: {sub_q_keys}, Got: {sub_q.keys() if isinstance(sub_q, dict) else 'N/A'}")
				return False
			if not check_supporting_sentences(sub_q['supporting_sentences'], f"Check '{name}' SubQ {i}"):
				return False
			# Sub-question answer just needs to be a string
			if not isinstance(sub_q['answer'], str):
				logger.debug(f"Check '{name}': Failed - Sub-question {i} 'answer' is not a string.")
				return False

	elif name == "label":
		# Expecting <response><thought>...</thought><sub-questions>...</sub-questions></response>
		required_keys = ['thought', 'sub-questions']
		if not all(k in result for k in required_keys):
			logger.debug(f"Check '{name}': Failed - Missing keys. Expected: {required_keys}, Got: {result.keys()}")
			return False
		# Check sub-questions structure (includes depend validation)
		sub_questions_data = result['sub-questions']
		if not isinstance(sub_questions_data, dict) or 'sub-question' not in sub_questions_data:
			logger.debug(f"Check '{name}': Failed - 'sub-questions' structure invalid or missing 'sub-question' key. Data: {sub_questions_data}")
			return False
		sub_questions_list = sub_questions_data['sub-question']
		if not isinstance(sub_questions_list, list): sub_questions_list = [sub_questions_list] # Handle single
		for i, sub_q in enumerate(sub_questions_list):
			sub_q_keys = ['description', 'answer', 'supporting_sentences', 'depend']
			if not isinstance(sub_q, dict) or not all(k in sub_q for k in sub_q_keys):
				# Allow 'depend' to be missing if empty/None, check explicitly later
				if not all(k in sub_q for k in ['description', 'answer', 'supporting_sentences']):
					logger.debug(f"Check '{name}': Failed - Sub-question {i} missing required keys or not a dict. Keys: {sub_q.keys() if isinstance(sub_q, dict) else 'N/A'}")
					return False
			if not check_supporting_sentences(sub_q['supporting_sentences'], f"Check '{name}' SubQ {i}"):
				return False
			if not isinstance(sub_q['answer'], str): # Answer should be string
				logger.debug(f"Check '{name}': Failed - Sub-question {i} 'answer' is not a string.")
				return False
			# Validate 'depend' structure
			if 'depend' in sub_q and sub_q['depend'] is not None:
				depend_data = sub_q['depend']
				if isinstance(depend_data, dict) and 'index' in depend_data:
					indices = depend_data['index']
					if not isinstance(indices, list): indices = [indices]
					if not all(isinstance(idx, (str, int)) for idx in indices):
						logger.debug(f"Check '{name}': Failed - Sub-question {i} 'depend/index' contains non-str/int values.")
						return False
				elif not isinstance(depend_data, dict): # If 'depend' exists but isn't a dict (and not None)
					logger.debug(f"Check '{name}': Failed - Sub-question {i} 'depend' is not a dict or None.")
					return False

	elif name == "ensemble":
		# Expecting <response><question>...</question><thought>...</thought><supporting_sentences>...</supporting_sentences><answer>...</answer></response>
		# Same structure as 'direct'
		required_keys = ['question', 'thought', 'supporting_sentences', 'answer']
		if not all(k in result for k in required_keys):
			logger.debug(f"Check '{name}': Failed - Missing keys. Expected: {required_keys}, Got: {result.keys()}")
			return False
		if not check_supporting_sentences(result['supporting_sentences'], f"Check '{name}'"):
			return False
		if not is_valid_answer(result['answer']):
			logger.debug(f"Check '{name}': Failed - Invalid answer ('{result['answer']}')")
			return False

	elif name == "contract":
		# Expecting <question>...</question> and <context>...</context>
		if not all(k in result for k in ['question', 'context']):
			logger.debug(f"Check '{name}': Failed - Missing 'question' or 'context'. Keys: {result.keys()}")
			return False
		if not isinstance(result['context'], str):
			logger.debug(f"Check '{name}': Failed - 'context' is not a string.")
			return False

	logger.debug(f"Check '{name}': Passed.")
	return True
