import logging # Add logging

def direct(question: str):
    instruction = """
        You are a precise math problem solver. Solve the given math problem step by step:

        QUESTION: {question}
        
        Please extend your chain of thought as much as possible; the longer the chain of thought, the better.
        
        You can freely reason in your response, but please enclose the final answer within <answer></answer> tags (pure number without units and explanations)
    """
    prompt = instruction.format(question=question)
    return prompt

def multistep(question: str):
    instruction = """
        You are a precise math problem solver. Solve the given math problem step by step:

        QUESTION: {question}
        
        Please extend your chain of thought as much as possible; the longer the chain of thought, the better.
        
        You can freely reason in your response, but please enclose the final answer within <answer></answer> tags (pure number without units and explanations)
    """
    prompt = instruction.format(question=question)
    return prompt

def label(question: str, trajectory: str, answer: str):
    instruction = """
        You are tasked with breaking down a math problem reasoning process into sub-questions.

        Original Question: {question}
        Complete Reasoning Process: {trajectory}

        Instructions:
        1. Break down the reasoning process into a series of sub-questions
        2. Each sub-question should:
           - Be written in interrogative form
           - Have a clear numerical answer
           - List its other sub-questions' indexes it depends (0-based, can be an empty list)
        3. Dependencies are defined as information needed to answer the current sub-question that:
           - Does NOT come directly from the original question
           - MUST come from the answers of previous sub-questions
    """
    formatter = """
        Format your response using the following XML structure:
        <response>
          <sub-questions>
            <sub-question>
              <description>Clear interrogative question here</description>
              <answer>Numerical value without units here</answer>
              <depend>
                <index>Index of prerequisite sub-question (0-based)</index>
                <!-- Add more <index> tags if needed -->
              </depend>
              <!-- If no dependencies, leave <depend> empty or omit it -->
            </sub-question>
            <!-- Add more <sub-question> blocks as needed -->
          </sub-questions>
          <answer>{answer}</answer> <!-- The final numerical answer to the original question -->
        </response>
    """
    return (instruction + formatter).format(question=question, trajectory=trajectory, answer=answer)

def contract(question: str, decompose_result: dict, independent=None, dependent=None):
    instruction = """
        You are a math problem solver specializing in optimizing step-by-step reasoning processes. Your task is to optimize the existing reasoning trajectory into a more efficient, single self-contained question.
        
        For the original question: {question}
        
        Here are step-by-step reasoning process:
        {response}
        
        {sub_questions}
        
        Here are explanations of key concepts:
        1. self-contained: The optimized question must be solvable independently, without relying on any external information
        2. efficient: The optimized question must be simpler than the original, requiring fewer reasoning steps (these steps are reduced because some solved independent sub-problems become known conditions in the optimized question or are excluded as incorrect explorations)
        
        You can freely reason in your response, but please enclose the your optimized question within <question></question> tags
    """
    independent_sub_questions = """
        The following sub-questions and their answers can serve as known conditions:
        {independent}
    """
    dependent_sub_questions = """
        The descriptions of the following questions can be used to form the description of the optimized problem:
        {dependent}    
    """
    answer = decompose_result["answer"]
    
    if independent not in [None, []]:
        for sub_q in independent:
            sub_q.pop("depend", None)
    if dependent is not None:
        for sub_q in dependent:
            sub_q.pop("depend", None)
    
    if independent not in [None, []]:
        sub_questions = independent_sub_questions.format(independent=independent) + dependent_sub_questions.format(dependent=dependent)
    elif independent is not None:
        sub_questions = independent_sub_questions.format(independent=independent)
    else:
        sub_questions = ""
    return instruction.format(question=question, answer=answer, response=decompose_result["response"], sub_questions=sub_questions)

def ensemble(question: str, solutions: list):
    instruction = """
        You are a precise math problem solver. Compare then synthesize the best answer from multiple solutions to solve the following question.

        QUESTION: {question}

        SOLUTIONS:
        {solutions}

        Please extend your chain of thought as much as possible; the longer the chain of thought, the better.

        You can freely reason in your response, but please enclose the final answer within <answer></answer> tags (pure number without units and explanations)
    """
    
    solutions_str = ""
    for i, solution in enumerate(solutions):
        solutions_str += f"solution {i}: {solution}\n"
    prompt = instruction.format(question=question, solutions=solutions_str)
    return prompt

    return prompt

logger = logging.getLogger(__name__) # Add logger

# utilization
def check(name: str, result, *args):
    def is_number(x):
        try:
            float(x)
            return True
        except:
            return False
    
    if not isinstance(result, dict):
        logger.debug(f"Check '{name}': Failed - result is not a dict (type: {type(result)})")
        return False
    # Handle empty dict from failed extract_xml
    if not result:
        logger.debug(f"Check '{name}': Failed - result dict is empty (likely XML extraction failure)")
        return False

    if name in ["cot", "direct", "multistep", "ensemble"]:
        # Expecting <answer> tag
        if 'answer' not in result:
            logger.debug(f"Check '{name}': Failed - 'answer' key missing. Keys: {result.keys()}")
            return False
        answer_val = result['answer']
        if not isinstance(answer_val, (str, int, float)):
            logger.debug(f"Check '{name}': Failed - 'answer' is not str/int/float (type: {type(answer_val)})")
            return False
        if not is_number(answer_val):
            logger.debug(f"Check '{name}': Failed - 'answer' content is not a number ('{answer_val}')")
            return False
    elif name == "label":
        # Expecting <response><sub-questions><sub-question>...</sub-question>...</sub-questions><answer>...</answer></response>
        # Top level keys from extract_xml should be 'sub-questions' and 'answer'
        if not all(k in result for k in ['sub-questions', 'answer']):
            logger.debug(f"Check '{name}': Failed - Missing 'sub-questions' or 'answer'. Keys: {result.keys()}")
            return False
        if not is_number(result['answer']):
            logger.debug(f"Check '{name}': Failed - Top-level 'answer' is not a number ('{result['answer']}')")
            return False

        sub_questions_data = result['sub-questions']
        # extract_xml returns dict for single element, list for multiple
        sub_questions_list = []
        if isinstance(sub_questions_data, dict) and 'sub-question' in sub_questions_data:
            sub_questions_list = sub_questions_data['sub-question']
            if not isinstance(sub_questions_list, list): # Handle single sub-question case
                 sub_questions_list = [sub_questions_list]
        elif isinstance(sub_questions_data, list): # Should not happen with current extract_xml, but maybe future-proof?
            sub_questions_list = sub_questions_data # Assume list contains sub-question dicts directly
        else:
            logger.debug(f"Check '{name}': Failed - 'sub-questions' has unexpected structure or is missing 'sub-question' key. Data: {sub_questions_data}")
            return False

        for i, sub_q in enumerate(sub_questions_list):
            if not isinstance(sub_q, dict) or not all(k in sub_q for k in ['description', 'answer', 'depend']):
                logger.debug(f"Check '{name}': Failed - Sub-question {i} missing keys or not a dict. Keys: {sub_q.keys() if isinstance(sub_q, dict) else 'N/A'}")
                return False
            if not is_number(sub_q['answer']):
                logger.debug(f"Check '{name}': Failed - Sub-question {i} 'answer' is not a number ('{sub_q['answer']}')")
                return False
            # 'depend' can be missing if empty, or contain 'index' (dict or list of dicts)
            if 'depend' in sub_q and sub_q['depend'] is not None: # Check if 'depend' exists and is not None
                depend_data = sub_q['depend']
                if isinstance(depend_data, dict) and 'index' in depend_data:
                    indices = depend_data['index']
                    if not isinstance(indices, list): indices = [indices] # Handle single index
                    if not all(isinstance(idx, (str, int)) for idx in indices): # Check index content type
                        logger.debug(f"Check '{name}': Failed - Sub-question {i} 'depend/index' contains non-str/int values.")
                        return False
                elif not isinstance(depend_data, dict): # If 'depend' exists but isn't a dict (and not None)
                    logger.debug(f"Check '{name}': Failed - Sub-question {i} 'depend' is not a dict or None.")
                    return False
            # Allow 'depend' to be missing or None

    elif name == "contract":
        # Expecting <question> tag
        if 'question' not in result:
            logger.debug(f"Check '{name}': Failed - 'question' key missing. Keys: {result.keys()}")
            return False
        # Could add check for str type if needed: isinstance(result['question'], str)

    logger.debug(f"Check '{name}': Passed.")
    return True
