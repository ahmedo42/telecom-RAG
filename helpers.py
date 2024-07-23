import pandas as pd 
import re 
import random
import json
import requests
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate

RAGLESS_PROMPT = """ 
        answer the following mcq considering the options
        the question is: 

        {q}

        ##############
        the options are:
        1) Option 1: {o1}
        2) Option 2: {o2}
        3) Option 3: {o3}
        4) Option 4: {o4}
        5) Option 5: {o5}
        ################
        
        Output: The correct option is: 
        """

RAG_PROMPT = """
        answer the following mcq considering the options
        the question is: 

        {q}

        considering the following context: {context}
        
        ##############
        the options are:
        1) Option 1: {o1}
        2) Option 2: {o2}
        3) Option 3: {o3}
        4) Option 4: {o4}
        5) Option 5: {o5}
        ################
        Output: The correct option is: 
        """



def make_submission(answers, task_name='Phi-2'):
    # given a list of Q/A pairs and model name, generate a submission file
    sub_file = pd.read_csv('SampleSubmission.csv')
    sub_file['Answer_ID'] = answers
    sub_file['Task'] = [task_name] * len(answers)
    sub_file.to_csv('submission.csv',index=False) 


def eval(gt,answers):
    # evaluate the LLM answers against an eval set 
    matching = 0 
    for e1,e2 in zip(gt,answers):
        matching += e1 == e2 
    return matching / len(answers)

def _falcon_helper(text_prompt):

    url = 'https://a7tncd274b.execute-api.eu-west-3.amazonaws.com/InitialStage/Falcon7B'
    # Prepare the data payload as a dictionary
    data = {
        "inputs": f"{text_prompt}",
        "parameters": {
            "max_new_tokens": 50,
            "return_full_text": False,
            "do_sample": True,
            "top_k": 50
        }
    }

    # Convert the dictionary to a JSON string
    json_data = json.dumps(data)

    # Set the appropriate headers for a JSON payload
    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, data=json_data, headers=headers)
        return json.loads(json.loads(response.text)['body'])[0]['generated_text']
    except Exception as e:
        print("An error occurred:", e)
        return ""
    
def preprocess_file(txt_path, max_samples=None):
    all_questions = []
    all_options = []
    answers = []

    with open(txt_path) as f:
        data = json.load(f)
        
        # Get a list of keys
        keys = list(data.keys())
        
        # If max_samples is specified, sample the keys
        if max_samples:
            keys = random.sample(keys, max_samples)
        
        for key in keys:
            entry = data[key]
            options = [entry.get(f'option {i}') for i in range(1, 6)]
            
            # Remove unwanted keys if 'train' is in the file path
            # save the answer for later evaluation 
            if 'train' in txt_path:
                answers.append(entry['answer'])
                keys_to_remove = ('explanation', 'answer', 'category')
                for k in keys_to_remove:
                    entry.pop(k, None)  # Use pop with default value to avoid KeyError
            
            all_questions.append(entry['question'])
            all_options.append(options)

    if 'train' in txt_path:
        return all_questions, all_options, answers
    
    else:
        return all_questions, all_options

def inference(all_questions, all_options , rag_system= None ,model='phi'):
    responses = []

    if rag_system:
        prompt = PromptTemplate(RAG_PROMPT)

    else:
        prompt = PromptTemplate(RAGLESS_PROMPT)
        llm = HuggingFaceLLM(model_name="microsoft/phi-2",tokenizer_name='microsoft/phi-2')

    for i,(question, options) in enumerate(zip(all_questions, all_options)):
        query = prompt.partial_format(q=question,
                              o1 = options[0],
                              o2 = options[1],
                              o3 = options[2],
                              o4 = options[3],
                              o5 = options[4])
        
        print(f"processing question {i}")
        if rag_system:
            print(f"question: {question}")
            context = rag_system.retrieve(question)
            query = query.partial_format(context = context)
            print(f"query: {query}")
            response = rag_system.answer(query)

        else:
            if model == 'phi':
                response = llm.complete(query)
            else:
                response = _falcon_helper(query)
        responses.append(response)

    return responses


def parse(responses,model='phi'):
    # extract the answer IDs from the llm responses
    # current approach is flawed, some more prompt engineering is required 
    # current approach is to extract the first occurance of Option {answer}, and randomly fill it if it's missing
    random.seed(42)
    answers = []
    for resp in responses:
        pattern = re.compile('The correct option is (\d+)')
        if model == 'phi':
            resp = resp.text

        match = pattern.search(resp)
        
        if match:
            answers.append(int(match.group(1)))
        else:
            answers.append(random.randint(1,5))
    return answers

