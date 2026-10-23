def subquery_prompt_v0(query_str, context_str, prev_reasoning):
    SUBQUERY_PROMPT = f"""role="system"
    The original question is as follows: {query_str} 
    We have an opportunity to answer some, or all of the question from a knowledge source. Context information for the knowledge source is provided below, as well as previous reasoning steps. 
    Given the context and previous reasoning, return a question that can be answered from the context. 
    This question can be the same as the original question, or represent a subcomponent. 
    It should not be irrelevant to the original question. 
    If no further information can be extracted, return 'None'.  

    Examples:  
    Question: How many Grand Slam titles does the winner of the 2020 Australian Open have? 
    Knowledge source context: Provides names of the winners of the 2020 Australian Open 
    Previous reasoning: None 
    Next question: Who was the winner of the 2020 Australian Open? 

    Question: How many Grand Slam titles does the winner of the 2020 Australian Open have? 
    Knowledge source context: Includes biographical info for each winner 
    Previous reasoning: 
    - Who was the winner of the 2020 Australian Open? 
    - The winner was Novak Djokovic. 
    Next question: How many Grand Slam titles does Novak Djokovic have?  

    Current Input:  Question: {query_str} 
    Knowledge source context: {context_str} 
    Previous reasoning: {prev_reasoning}

    assistant:
    % Output: <decomposed sub-question> OR 'None'
    """
    return SUBQUERY_PROMPT

def subquery_prompt_v1(query_str, context_str, prev_reasoning, missing_info):
    SUBQUERY_PROMPT_IMPROVED = f"""You are a Query Evolution Agent. Your goal is to decompose a complex user question into a specific, actionable sub-query to retrieve missing information from a Knowledge Graph.

The original question is: "{query_str}"

We have already performed some reasoning.
Current Known Information (Context):
{context_str}

History of Reasoning Steps (Q&A):
{prev_reasoning}

CRITICAL: The Reflector Agent has identified the following MISSING INFORMATION needed to answer the main question:
{missing_info}

Task:
Based on the "Missing Information" and "History", formulate the **NEXT** single sub-question to retrieve this missing info.
- The sub-question must be specific.
- It should act as a search query for the next hop.
- If the "Missing Information" implies we have enough to answer the main question, return 'None'.

Sub-question: 
"""
    return SUBQUERY_PROMPT_IMPROVED