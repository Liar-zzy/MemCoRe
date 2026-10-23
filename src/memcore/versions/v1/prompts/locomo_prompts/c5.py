def answer_c5_v5(context, question, answer_tmp):
    CATEGORY_5_PROMPT = f"""
You are a "Context Matcher" AI. Your job is NOT to answer the question faithfully, but to detect if a specific TOPIC exists in the memory, regardless of the context or people involved.

[Retrieved Memory]:
{context}

[Question (WARNING: Contains Unreliable Information)]:
{question}
*(Note: The question often names the WRONG person, WRONG time, or WRONG intent. Ignore these errors. Focus ONLY on the core object/activity asked about.)*

[Candidate Options]:
Option A: {answer_tmp[0]}
Option B: {answer_tmp[1]}

[MATCHING PROTOCOL - AGGRESSIVE MODE]:
1. **Identify the "Real" Option**: One option is a specific detail (e.g., "old car", "Kyle", "painting of a castle"). The other is "Not mentioned". Focus entirely on the specific detail.

2. **Search for "Concept Traces"**: Scan the memory for the *Objects* or *Actions* in that specific option.
   - Case: Question asks about "Maria's child". Option is "Kyle". Memory mentions "John's son Kyle". -> **MATCH**. (Select "Kyle").
   - Case: Question asks about "Luxury store donation". Option is "old car". Memory says "donated a vehicle to a shop". -> **MATCH**. (Select "old car").
   - Case: Question asks about "Maria running for office". Option is "impact in community". Memory says "John saw the impact... through politics". -> **MATCH**. (Select the impact option).

3. **Override the "Who/When/Why"**:
   - If the text mentions the *Event* but attributes it to the wrong person -> **IT IS A MATCH**.
   - If the text mentions the *Object* but describes it slightly differently (e.g., "castle" vs "castle on a hill") -> **IT IS A MATCH**.

4. **The "Not Mentioned" Threshold**:
   - You may ONLY select "Not mentioned in the conversation" if the *Keywords* of the specific option (e.g., "Kyle", "Car", "Certificate", "Castle") appear **NOWHERE** in the text.

[Task]:
Select the option that matches the text traces. Output ONLY the option text.

Short answer:
    """
    return CATEGORY_5_PROMPT

def answer_c5_v4(context, question, answer_tmp):
    CATEGORY_5_PROMPT = f"""
Based on the [Retrieved Memory] below, answer the following [question].

[Retrieved Memory]:
{context}

[Question]:
{question}

[Options]:
Option A: {answer_tmp[0]}
Option B: {answer_tmp[1]}

[CRITICAL INSTRUCTION - READ CAREFULLY]:
Your goal is to find the "Best Possible Match" based on the Event/Action described, even if the names don't match.

1. **Ignore Entity/Speaker Mismatches**: The question might have the wrong person's name (e.g., asking what "Maria" did, but the text says "John" did it).
   - IF the specific activity/event (e.g., "beginner's yoga class", "run for office") appears in the text, YOU MUST SELECT IT.
   - Do NOT select "Not mentioned" just because the names are swapped.

2. **Semantic Matching**: Match the meaning, not just exact words.
   - Example: Text says "painting of a castle" -> Option says "painting of a castle on a hill" -> THIS IS A MATCH.
   - Example: Text says "Mon and Wed" -> Option says "A few times a week" -> THIS IS A MATCH.

3. **"Not Mentioned" is the Last Resort**: Only select "Not mentioned in the conversation" if the *activity* or *topic* is completely absent from the text for ALL characters.

[Task]:
Output ONLY the text of the matching option.

Short answer:
    """
    return CATEGORY_5_PROMPT

def answer_c5_v3(context, question, answer_tmp):
    CATEGORY_5_PROMPT = f"""
Based on the [Retrieved Memory] below, answer the following [question].

[Retrieved Memory]:
{context}

[Question]:
{question}

[Options]:
Option A: {answer_tmp[0]}
Option B: {answer_tmp[1]}

[Evaluation Rules - READ CAREFULLY]:
1. **Goal**: Identify if the memory supports one of the specific options.
2. **Inference Allowed**: If the memory describes specific instances (e.g., "Mondays and Wednesdays"), and the option summarizes it (e.g., "A few times a week"), this is a MATCH. Select the specific option.
3. **Partial Match Allowed**: If the memory mentions a core object (e.g., "painting of a castle") and the option adds descriptive details (e.g., "painting of a castle on a hill"), treat this as a MATCH.
4. **"Not mentioned" Restriction**: ONLY select "Not mentioned in the conversation" if the topic is completely absent from the memory. If there is ANY semantic overlap, choose the specific detail option.

[Task]:
Think step-by-step internally:
1. Does the memory mention the topic?
2. Does the specific option loosely match the memory?
3. If yes, the answer is the specific option.

Finally, output ONLY the text of the selected option. Do not output the thinking process.

Short answer:
    """
    return CATEGORY_5_PROMPT

def answer_c5_v2(context, question, answer_tmp):
    CATEGORY_5_PROMPT = f"""
Based on the [Retrieved Memory] below, answer the following [question].

[Retrieved Memory]:
{context}

[Question]:
{question}

[Options]:
Option A: {answer_tmp[0]}
Option B: {answer_tmp[1]}

[Evaluation Rules - READ CAREFULLY]:
1. **Goal**: Identify if the memory supports one of the specific options.
2. **Inference Allowed**: If the memory describes specific instances (e.g., "Mondays and Wednesdays"), and the option summarizes it (e.g., "A few times a week"), this is a MATCH. Select the specific option.
3. **Partial Match Allowed**: If the memory mentions a core object (e.g., "painting of a castle") and the option adds descriptive details (e.g., "painting of a castle on a hill"), treat this as a MATCH.
4. **"Not mentioned" Restriction**: ONLY select "Not mentioned in the conversation" if the topic is completely absent from the memory. If there is ANY semantic overlap, choose the specific detail option.

[Task]:
Select the correct text from Option A or Option B.
Output ONLY the text of the correct option without any prefixes or reasoning.

Short answer:
    """
    return CATEGORY_5_PROMPT

def answer_c5_v1(context, question, answer_tmp):
    CATEGORY_5_PROMPT = f"""
    Based on the [Retrieved Memory] below, answer the following [question].
    
    [Retrieved Memory]:
    {context}

    [Question]:
    {question}

    Select the correct answer: {answer_tmp[0]} or {answer_tmp[1]}

    Short answer:
    """
    return CATEGORY_5_PROMPT


def answer_c5_v0(context, question, answer_tmp):
    CATEGORY_5_PROMPT = f"""
    Based on the context: {context}, answer the following question. {question} 
    
    Select the correct answer: {answer_tmp[0]} or {answer_tmp[1]}  
    Short answer:
    """
    return CATEGORY_5_PROMPT


