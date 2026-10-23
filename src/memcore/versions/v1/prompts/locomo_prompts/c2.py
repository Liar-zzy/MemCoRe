def answer_c2_v8(context, question):
    CATEGORY_2_PROMPT = f"""
Task: Analyze the [Question] intent and perform either **Fact Extraction** or **Temporal Calculation** based on the [Retrieved Memory].

[Retrieved Memory]:
{context}

[Question]: 
{question}

---
**STRICT ALGORITHMIC PROTOCOL:**

You must perform the analysis internally and output **ONLY** the final result string. Do not output any explanation or reasoning.

**INTERNAL EXECUTION STEPS (Do not output these):**

1.  **Locate Anchor Date**: Find the `[Time: ...]` tag in the memory. Use this as your calculation baseline.

2.  **Determine Intent & Calculate**:
    * **CASE A: Question asks for a Specific Date ("When...?")**
        * Identify the relative phrase in the text (e.g., "last Monday", "the week before").
        * **CRITICAL MATH**: You MUST calculate the exact date based on the Anchor Date.
            * *Example*: If Anchor is May 25 (Thursday) and text says "Sunday before", the answer is May 21.
            * **Prohibition**: Do NOT simply output the Anchor Date unless the event happened exactly on that day.
        * **Format**: `D Month YYYY` (e.g., 21 May 2023) or `Month YYYY`.

    * **CASE B: Question asks for Duration ("How long ago...?")**
        * Extract the exact time string mentioned (e.g., "10 years", "3 weeks").
        * **Prohibition**: Do NOT convert this to a year (e.g., do NOT convert "10 years ago" to "2013"). Output the text "10 years".

    * **CASE C: Question asks for Entity ("Who...?", "What...?")**
        * Extract the subject.
        * **Priority**: Prefer **Proper Nouns** (Capitalized names like "Max") over common nouns (like "a pet").

3.  **Handling Missing Info**:
    * If the answer is explicitly not in the text, output: "Not mentioned".

---
**Answer:**
"""
    return CATEGORY_2_PROMPT

def answer_c2_v7(context, question):
    CATEGORY_2_PROMPT = f"""
Task: Analyze the [Question] intent and perform either **Fact Extraction** or **Temporal Calculation** based on the [Retrieved Memory].

[Retrieved Memory]:
{context}

[Question]: 
{question}

---
**STRICT ALGORITHMIC PROTOCOL:**

1.  **INTENT CLASSIFICATION (Step 1)**:
    * Analyze the Question Word:
        * **Type A (Entity/Quantity)**: Questions starting with "What", "Where", "Which", "Who", "How many" (asking for count/duration), "How long".
        * **Type B (Timestamp)**: Questions starting with "When" or asking for a specific date/time.

2.  **EXECUTION PATHS (Choose ONE based on Intent)**:

    * **PATH A: IF Type A (Entity/Quantity/Location)**:
        * **Action**: Extract the specific short phrase, number, duration string (e.g., "3 days"), or entity requested.
        * **Constraint**: Do **NOT** output the `[Time: ...]` tag or calculate dates. Just answer the fact.
        * *Example Logic*: If asking "How many days?", extract "19 days". If asking "Where?", extract the location.

    * **PATH B: IF Type B (When/Date)**:
        * **Action**: Perform **Temporal Normalization**.
        * **Anchor**: Locate `[Time: ...]` tag as `Reference_Date`.
        * **Calculation**: `Answer` = `Reference_Date` + `Relative Offset` (e.g., "yesterday", "last year").
        * **Ban List**: DO NOT output "last week", "yesterday", etc. Output the calculated **Absolute Date**.
        * **Format**: Year (e.g., "2023"), Month Year (e.g., "May 2023"), or Day Month Year.

3.  **FINAL OUTPUT FORMAT**:
    * Output **ONLY** the result phrase or calculated date.
    * No complete sentences.

---
**Answer:**
"""
    return CATEGORY_2_PROMPT
def answer_c2_v6(context, question):

    CATEGORY_2_PROMPT = f"""
Task: **Temporal Normalization**. Convert the relative time in the memory into an **Absolute Date**.

[Retrieved Memory]:
{context}

[Question]: 
{question}

---
**STRICT ALGORITHMIC PROTOCOL:**

1.  **ANCHOR ACQUISITION**:
    * Locate the `[Time: Day Month, Year]` tag at the beginning of the text.
    * Set this as `Current_Reference_Date`.

2.  **OFFSET DETECTION & CALCULATION**:
    * Scan the text for relative time expressions (Target List: "yesterday", "last week", "last month", "last year", "ago").
    * **IF** a target expression is found:
        * Perform arithmetic: `Final_Date` = `Current_Reference_Date` + `Relative_Offset`.
        * (e.g., May - 1 month = April).
    * **IF** no relative expression is found:
        * `Final_Date` = The explicit date mentioned in text.

3.  **OUTPUT FIREWALL (The "Ban List"):**
    * The output **MUST NOT** contain any words from the Target List.
    * **VIOLATION CHECK**: If your answer is "last month", "last year", or "yesterday", it is **WRONG**.
    * **CORRECTION**: You must output the calculated `Month Year` or `Year` or `Date`.

4.  **FORMATTING RULES**:
    * **For Year-level offset ("last year"):** Output **Year Only** (e.g., "2022"). Do NOT invent a day.
    * **For Month-level offset ("last/next month"):** Output **Month Year** (e.g., "December 2022").
    * **For Day-level offset ("yesterday"):** Output **Day Month Year**.

---
**Answer (Calculated Date Only):**
"""
    return CATEGORY_2_PROMPT

def answer_c2_v5(context, question):

    CATEGORY_2_PROMPT = f"""
Task: Perform strict **Temporal Calculation** to determine the actual event date based on the [Retrieved Memory].

[Retrieved Memory]:
{context}

[Question]: 
{question}

---
**ALGORITHMIC INSTRUCTIONS (Zero-Shot Reasoning):**

1.  **DEFINE VARIABLES**:
    * `T_Anchor`: The date found in the `[Time: ...]` tag. (This is when the speaker spoke).
    * `Offset`: Relative words in the text (e.g., "yesterday" = -1 day, "last week" = -7 days, "next month" = +1 month).

2.  **CALCULATION RULE (The "Not T_Anchor" Rule)**:
    * **IF** the text contains `Offset` words like "yesterday", "last night", "a week ago":
        * **YOU MUST SUBTRACT/ADD** time from `T_Anchor`.
        * **CRITICAL**: The Answer is **ALMOST NEVER** `T_Anchor` itself.
        * *Logic*: If `T_Anchor` is May 8 and text says "yesterday", Answer MUST be May 7.

3.  **SCOPE ADJUSTMENT**:
    * **Specific Day**: If text says "yesterday" or a specific weekday, output the specific date (Day Month Year).
    * **General Timeframe**: If text says "last week" or "recently", output a range (e.g., "The week before [Date]").
    * **Month Level**: If text says "this month" or "next month", output only the **Month and Year** (e.g., "July 2023"), DO NOT include the specific day from `T_Anchor`.

4.  **OUTPUT FORMAT**:
    * Return ONLY the calculated result.

---
**Answer:**
"""
    return CATEGORY_2_PROMPT
    
def answer_c2_v4(context, question):

    CATEGORY_2_PROMPT = f"""
Based on the [Retrieved Memory] below, write an answer in the form of **a short phrase** for the following [Question], not a sentence. Answer with exact words from the [Retrieved Memory] whenever possible.
For questions that require answering a date or time, strictly follow the format \"15 July 2023\" and provide a specific date whenever possible. For example, if you need to answer \"last year,\" give the specific year of last year rather than just saying \"last year.\" Only provide one year, date, or time, without any extra responses.
If the question is about the duration, answer in the form of several years, months, or days.

[Retrieved Memory]:
{context}

[Question]: 
{question}

---
**LOGIC RULES:**

1. **TIME CALCULATION (Highest Priority)**:
   - **Anchor**: Locate the `[Time: Day Month, Year]` tag at the start of the memory.
   - **Calculate**: Convert relative words ("yesterday", "last year", "next month") into **Absolute Dates** using the Anchor.
   - **Formula**: `Answer` = `Anchor Date` +/- `Relative Offset`.
   - *Prohibition*: NEVER output "last week", "yesterday", or "last year".

2. **CONCISE FORMAT**:
   - Output **ONLY** the result (date or entity).
   - **No Sentences**: Delete "It was...", "The date is...".

---
**CALCULATION DEMONSTRATIONS (Structure matched to input):**

**Case 1: Relative Day ("Yesterday")**
*Input Context:* "(Score: 0.88) [Time: 15 Jan, 2023] [Content]: Speaker Bob: I submitted the report yesterday. [Context]: Bob finished his work."
*Question:* When did Bob submit the report?
*Logic:* 15 Jan - 1 day = 14 Jan.
*Answer:* 14 Jan, 2023

**Case 2: Relative Year ("Last Year")**
*Input Context:* "(Score: 0.92) [Time: 10 May, 2023] [Content]: Speaker Alice: I visited Paris last year and it was lovely. [Context]: Alice recalls her trip."
*Question:* When did Alice visit Paris?
*Logic:* 2023 - 1 year = 2022.
*Answer:* 2022

**Case 3: Relative Month ("Next Month")**
*Input Context:* "(Score: 0.75) [Time: 20 June, 2023] [Content]: Speaker Charlie: We are launching the product next month. [Context]: The team plans a launch."
*Question:* When is the product launch?
*Logic:* June + 1 month = July.
*Answer:* July 2023

---
**Your Answer:**
"""
    return CATEGORY_2_PROMPT

def answer_c2_v3(context, question):

    CATEGORY_2_PROMPT = f"""
Based on the [Retrieved Memory] below, write an answer in the form of **a short phrase** for the following [Question], not a sentence. Answer with exact words from the [Retrieved Memory] whenever possible.
For questions that require answering a date or time, strictly follow the format \"15 July 2023\" and provide a specific date whenever possible. For example, if you need to answer \"last year,\" give the specific year of last year rather than just saying \"last year.\" Only provide one year, date, or time, without any extra responses.
If the question is about the duration, answer in the form of several years, months, or days.

[Retrieved Memory]:
{context}

[Question]: 
{question}

Answer:
"""
    return CATEGORY_2_PROMPT

def answer_c2_v2(context, question):
    CATEGORY_2_PROMPT = f"""
Task: Determine the specific absolute date, timeframe, or short fact based on the [Context].

[Retrieved Memory]:
{context}

[Question]: 
{question}

---
**LOGIC RULES:**

1. **TIME CALCULATION (Highest Priority)**:
   - **Anchor**: Locate the `[Time: Day Month, Year]` tag at the start of the memory.
   - **Calculate**: Convert relative words ("yesterday", "last year", "next month") into **Absolute Dates** using the Anchor.
   - **Formula**: `Answer` = `Anchor Date` +/- `Relative Offset`.
   - *Prohibition*: NEVER output "last week", "yesterday", or "last year".

2. **CONCISE FORMAT**:
   - Output **ONLY** the result (date or entity).
   - **No Sentences**: Delete "It was...", "The date is...".

---
**CALCULATION DEMONSTRATIONS (Structure matched to input):**

**Case 1: Relative Day ("Yesterday")**
*Input Context:* "(Score: 0.88) [Time: 15 Jan, 2023] [Content]: Speaker Bob: I submitted the report yesterday. [Context]: Bob finished his work."
*Question:* When did Bob submit the report?
*Logic:* 15 Jan - 1 day = 14 Jan.
*Answer:* 14 Jan, 2023

**Case 2: Relative Year ("Last Year")**
*Input Context:* "(Score: 0.92) [Time: 10 May, 2023] [Content]: Speaker Alice: I visited Paris last year and it was lovely. [Context]: Alice recalls her trip."
*Question:* When did Alice visit Paris?
*Logic:* 2023 - 1 year = 2022.
*Answer:* 2022

**Case 3: Relative Month ("Next Month")**
*Input Context:* "(Score: 0.75) [Time: 20 June, 2023] [Content]: Speaker Charlie: We are launching the product next month. [Context]: The team plans a launch."
*Question:* When is the product launch?
*Logic:* June + 1 month = July.
*Answer:* July 2023

---
**Your Answer:**
"""
    return CATEGORY_2_PROMPT

def answer_c2_v1(context, question):
    CATEGORY_2_PROMPT = f"""
Task: Determine the specific date, timeframe, or short fact based on the [Context].

[Context] (Note the `[Time: ...]` tags):
{context}

[Question]: 
{question}

---
**LOGIC RULES:**

1. **TIME CALCULATION (Crucial)**:
   - Identify the `[Time: ...]` tag associated with the event. Treat this as the **Anchor Date**.
   - **MANDATORY**: Convert relative terms (e.g., "last year", "yesterday", "next month", "this month") into **Absolute Dates** based on the Anchor Date.
   - **Formula**: `Answer` = `Anchor Date` + `Relative Offset`.
   - *Prohibition*: NEVER output "last week/year". Output the calculated year or date range (e.g., "May 2022" or "The week of June 5th").

2. **CONCISE FORMAT**:
   - Output **ONLY** the answer entity or date.
   - **No Sentences**: Do not include subjects or verbs (e.g., Delete "It was...", "She went...").
   - **No Explanations**: Just the result.

Answer:
"""
    return CATEGORY_2_PROMPT

def answer_c2_v0(context, question):
    CATEGORY_2_PROMPT = f"""
    Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
    Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.   

    Question: {question} 
    Short answer:
    """
    return CATEGORY_2_PROMPT