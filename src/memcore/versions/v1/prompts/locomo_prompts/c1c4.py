def answer_c1_c4_v11(context, question):
    CATEGORY_C1_C4_PROMPT = f"""
Based on the [Retrieved Memory] below, answer the following [question].

[Retrieved Memory]:
{context}

[Question]: 
{question}

---
**TASK: STRICT DATA EXTRACTION**
Answer the question directly using the information from the memory.

**OUTPUT FORMAT RULES (STRICT):**
1.  **NO SENTENCES:** Output ONLY the answer entities or phrase. No "The answer is...", no verbs unless describing a method.
2.  **LISTS:** Comma-separated.
3.  **COUNTS:** Output a single Arabic numeral (e.g., "2").
4.  **BOOLEAN:** "Yes" or "No".
5.  **UNKNOWN:** Output "Unknown" ONLY if the information is completely absent.

**SEARCH & EXTRACTION LOGIC (INTERNAL OVERRIDES):**

* **RULE 1: MAXIMIZE SPECIFICITY (Fixes "Red Car" vs "Ferrari")**
    * Target **Proper Nouns** (Brands, Cities, Names) over generic nouns.
    * *Example:* If text says "bought a luxury car, a Ferrari", extract "Ferrari".
    * *Example:* If text says "octopus item", look closer for the object name (e.g., "guitar").

* **RULE 2: AGGRESSIVE SCANNING (Fixes "Unknown" & Missing Counts)**
    * **Do not stop at the first match.** You MUST scan the **ENTIRE** text to find all list items or count all occurrences.
    * **Look for Time Markers:** If asking about "future" or "plans", look for keywords like "planning to", "next", "after".
    * **Look for Synonyms:** The answer might not use the exact words in the question (e.g., "tour" vs "concert").

* **RULE 3: HANDLING "WHY" & "HOW" (Category 4)**
    * Extract the **core reason** or **feeling** as a short fragment.
    * *Acceptable:* "To relax", "In awe", "Freedom".
    * *Forbidden:* Full sentences like "He goes there to relax."

**Your Answer:**
    """
    return CATEGORY_C1_C4_PROMPT
def answer_c1_c4_v10(context, question):
    CATEGORY_C1_C4_PROMPT = f"""
Based on the [Retrieved Memory] below, answer the following [question].

[Retrieved Memory]:
{context}

[Question]: 
{question}

---
**TASK: HIGH-PRECISION DATA EXTRACTION**
You are a strict query-answering engine. Adapt your output logic based on the specific **Question Intent**.

**Thinking Process (Execute internally):**
1.  **Analyze Intent:** Is it asking for an Entity (Who/Where), a State (Status/Identity), a Reason (Why), or a Verification (Yes/No)?
2.  **Locate Evidence:** Find the exact sentence in memory.
3.  **Check Constraints:** For "Both/Common" questions, ensure the answer applies to ALL subjects.
4.  **Format Output:** Apply the strict grammar rules below.

---
**OUTPUT MODES & GRAMMAR RULES:**

**MODE A: ENTITIES & ATTRIBUTES (Who, Where, Which, What Identity/Status)**
* **Target:** Concrete nouns, names, locations, or specific statuses (e.g., "Single", "Transgender woman").
* **Grammar:** Noun phrase ONLY. **NO verbs.**
* **Fix:** If asking "What is her identity?", output "Transgender woman" (NOT "She is a...").
* **Fix:** If asking "Relationship status?", output "Single" (NOT "She is single").

**MODE B: REASONING & DESCRIPTIONS (Why, How, What ... do/make/feel)**
* **Target:** Explanations, feelings, or method.
* **Grammar:**
    * **For "Why":** Start DIRECTLY with the cause (e.g., "Lost his job"). **FORBIDDEN:** Do not repeat "He did X because...".
    * **For "How (Method)":** Start with a Verb + ing or By + Noun (e.g., "By practicing daily").
    * **For "How (Feeling)" / "What make him":** Output the **Adjective** or **Emotion** (e.g., "Happy", "In awe").
* **Length Limit:** Max 15 words. Keep it fragments.

**MODE C: VERIFICATION (Did, Do, Is, Can)**
* **Target:** Boolean check.
* **Output:** **"Yes"** or **"No"** ONLY.
* **Exception:** If the answer is ambiguous, add a max 3-word qualifier (e.g., "Yes, partially").

**MODE D: TIME & DURATION (When, How long)**
* **Target:** Timepoint or Timespan.
* **Differentiation:**
    * **"When":** Output Date/Time (e.g., "19 Oct 2023").
    * **"How long":** Output Duration (e.g., "Six months", "2 hours").

---
**CRITICAL NEGATIVE CONSTRAINTS:**
1.  **NO REPETITION:** Never repeat the subject or verb of the question. (Q: "Why did she cry?" -> A: "She cried because..." [WRONG] -> A: "Due to sadness" [RIGHT]).
2.  **NO LIST FLUFF:** Do not use "and", "also", "as well as" to connect long sentences. Use commas for lists.
3.  **ANTI-HALLUCINATION:** If the exact answer is not explicitly in the text, output "Unknown". Do not guess generic answers like "home".

**Your Answer:**
"""
    return CATEGORY_C1_C4_PROMPT
def answer_c1_c4_v9(context, question):
    CATEGORY_C1_C4_PROMPT = f"""
Based on the [Retrieved Memory] below, answer the following [question].

[Retrieved Memory]:
{context}

[Question]: 
{question}

---
**TASK: ADAPTIVE INFORMATION EXTRACTION**
Your goal is to answer the question strictly based on the memory. You must adapt your output format based on the **Question Type**.

**MODE A: IF Asking "Who", "What (objects/events)", "Which", or for a List (Category 1 Style)**
* **Output:** A comma-separated list of **Concrete Entities** (Nouns/Phrases).
* **Constraint:** Prioritize Proper Nouns (e.g., "Sweden") over generic terms (e.g., "home country").
* **Anti-Hallucination:** Only list items explicitly mentioned. Do not infer lists from general descriptions.

**MODE B: IF Asking "Why", "How (feeling/manner)", or "Describe" (Category 4 Style)**
* **Output:** A short, precise **phrase** or **sentence fragment** explaining the reason or state.
* **Constraint:** Abstract concepts (feelings, reasons) ARE ALLOWED here.
* *Example:* Q: "How did she feel?" -> A: "In awe of the universe" (Acceptable).

**MODE C: IF Asking "Did", "Is", "Can" (Yes/No Questions)**
* **Output:** Start with **"Yes"** or **"No"**. You may add a 2-3 word verification.
* *Example:* "Yes", or "No, she stayed home".

**MODE D: IF Asking "When"**
* **Output:** Extract the specific date, time, or relative timeframe.
* *Example:* "19 October 2023".

---
**GLOBAL NEGATIVE CONSTRAINTS (Apply to ALL modes):**
1.  **NO VERBOSITY:** Do not start with "The answer is..." or "Melanie feels...". Go straight to the core information.
2.  **NO EXPLANATORY FLUFF:** Unless asking "Why", do not explain the answer.
3.  **STRICT GROUNDING:** If the specific answer is not in the text, output "Unknown".

**Thinking Process (Internal):**
1. Identify Question Type (Is it "Who/What", "Why/How", "Did/Is", or "When"?).
2. Select valid evidence from memory.
3. Format output according to the Mode selected above.

**Your Answer:**
"""
    return CATEGORY_C1_C4_PROMPT

def answer_c1_c4_v8(context, question):
    CATEGORY_C1_C4_PROMPT = f"""
Based on the [Retrieved Memory] below, answer the following [question].

[Retrieved Memory]:
{context}

[Question]: 
{question}

---
**SYSTEM TASK: STRICT DATA EXTRACTION**
You are functioning as a text extraction API. Your sole purpose is to parse the [Retrieved Memory] and extract specific entities that answer the [Question].

**GLOBAL CONSTRAINTS (MUST FOLLOW):**
1.  **NO CONVERSATION:** Do not write complete sentences. Do not use filler words like "Melanie likes..." or "She went to...".
2.  **NO TEMPORAL DATA:** Strictly remove all dates, times, and timestamps (e.g., convert "Pride Parade (July 3rd)" to "Pride Parade").
3.  **OUTPUT FORMAT:** Output **ONLY** a comma-separated list of entities (or a single number).

**CATEGORY-SPECIFIC LOGIC FILTERS:**

* **IF Asking for "Activities/Hobbies":**
    * **Target:** Physical actions or events (e.g., "Camping", "Pottery", "Swimming").
    * **FILTER OUT:** Abstract outcomes, feelings, or purposes.
    * *Rule:* "Creating family memories" is a PURPOSE, not an activity. REJECT IT.
    * *Rule:* "Destressing" is a FEELING. REJECT IT. Extract the method (e.g., "Running") instead.

* **IF Asking for "Books/Movies/Art":**
    * **Target:** Proper Titles (in quotes) or specific Genres.
    * **FILTER OUT:** Descriptions of physical appearance.
    * *Rule:* Do not output "book with a gold coin". If the title is missing, output "Unknown book".

* **IF Asking for "Counts" (How many...):**
    * **Target:** A single Arabic numeral (e.g., 3).
    * **Action:** Count the distinct occurrences in the text yourself and output only the digit.

* **IF Asking for "Likes/Interests":**
    * **Target:** Items explicitly described as favorites or interests.
    * **FILTER OUT:** Things the subject merely "did" but didn't explicitly say they "liked".

**FINAL CHECK:**
Does your answer contain dates? -> Remove them.
Is your answer a sentence? -> Convert to list.
Did you include abstract feelings? -> Remove them.

---
**Your Answer:**
"""
    return CATEGORY_C1_C4_PROMPT

def answer_c1_c4_v7(context, question):
    CATEGORY_C1_C4_PROMPT = f"""
Based on the [Retrieved Memory] below, answer the following [question].

[Retrieved Memory]:
{context}

[Question]: 
{question}

---
**TASK: ENTITY EXTRACTION ONLY**
Your goal is to extract **concrete nouns or short noun phrases** from the text that directly answer the question. 
Output must be a **comma-separated list**.

**CRITICAL RULES & CONSTRAINTS:**

1.  **CONCRETE NOUNS OVER DESCRIPTIONS (The "No Fluff" Rule):**
    * **EXTRACT:** Specific activities, objects, or event names (e.g., "camping", "pottery", "Pride parade", "Nothing is Impossible").
    * **REJECT:** Broad actions, emotional states, or summaries (e.g., remove "exploring nature", "creating memories", "acceptance", "learning to express herself").
    * *Example:* If text says "We went hiking to enjoy nature," extract "hiking," NOT "enjoying nature."

2.  **STRICT FORMATTING (Anti-Verbosity):**
    * **NO Sentences:** Never output a full sentence.
    * **NO Dates/Timestamps:** Do not include dates (e.g., "May 8, 2023") unless the question specifically asks "When."
    * **NO Counting/Bullet Points:** Output a flat list separated by commas (e.g., "Item A, Item B").

3.  **SEMANTIC PRECISION:**
    * **For "Activities/Hobbies":** Extract the *name* of the activity (e.g., "Running"), not the *purpose* (e.g., "to destress").
    * **For "Books/Art":** Extract the **Title** or **Genre** (e.g., "Charlotte's Web", "abstract art"). Do not describe the cover (e.g., "book with gold coin" -> INVALID).
    * **For "Likes" vs "Do":** If asking what someone *likes*, look for verbs like "loves", "enjoys", "obsessed with". Do not infer likes just because they did an activity.

4.  **HANDLING NUMBERS:**
    * If the question asks "How many", output a **single Arabic numeral** (e.g., "2"), implying you must count the distinct occurrences in memory yourself.

5.  **FALLBACK:**
    * If no specific entity is found, output "Unknown".

---
**Your Short Answer (Comma-separated list):**
"""
    return CATEGORY_C1_C4_PROMPT
def answer_c1_c4_v6(context, question):
    CATEGORY_C1_C4_PROMPT = f"""
Task: Extract **Specific Entities** based strictly on the [Retrieved Memory] to answer the [Question].

[Retrieved Memory]:
{context}

[Question]: 
{question}

---
**CRITICAL EXTRACTION RULES:**

1.  **HIERARCHY OF ACCURACY (Prioritize Specificity):**
    * **Priority 1 (Proper Nouns):** Look for capitalized names first.
    * **Priority 2 (Specific Roles/Groups):** If NO Proper Name exists, extract specific relationships or distinct groups.
    * **REJECT (Generic Vague Terms):** Strictly ignore vague placeholders like "home country", "a place", "someone", "people", "activities".

2.  **STRICT GROUNDING (Anti-Hallucination):**
    * **Do not infer:** If the text says "cooking", do NOT add "eating" unless stated.
    * **Exact Match:** The answer must be semantically present in the text.
    * **If missing:** If the exact answer is not in the text, output "Unknown".

3.  **CATEGORY-SPECIFIC RULES:**
    * **For "Who":** Prefer names. If asking "Who supports him?", distinct roles like "the manager" are acceptable.
    * **For "Where":** Must be a **Named Location** (e.g., "Mars") unless the question asks for a generic type. Reject "home planet" if a specific name appears.
    * **For "What":** Extract the **Direct Object** of the action.

4.  **NOISE FILTERING:**
    * Ignore `[Image: ...]` descriptions.
    * Strip introductory modifiers (e.g., remove "a pilot named").

---

**Your Short Answer:**
"""
    return CATEGORY_C1_C4_PROMPT
def answer_c1_c4_v5(context, question):
    CATEGORY_C1_C4_PROMPT = f"""
Task: Extract **Specific Entities** based strictly on the [Retrieved Memory] to answer the [Question].

[Retrieved Memory]:
{context}

[Question]: 
{question}

---
**CRITICAL EXTRACTION RULES:**

1.  **HIERARCHY OF ACCURACY (Prioritize Specificity):**
    * **Priority 1 (Proper Nouns):** Look for capitalized names first.
    * **Priority 2 (Specific Roles/Groups):** If NO Proper Name exists, extract specific relationships or distinct groups.
    * **REJECT (Generic Vague Terms):** Strictly ignore vague placeholders like "home country", "a place", "someone", "people", "activities".

2.  **STRICT GROUNDING (Anti-Hallucination):**
    * **Do not infer:** If the text says "cooking", do NOT add "eating" unless stated.
    * **Exact Match:** The answer must be semantically present in the text.
    * **If missing:** If the exact answer is not in the text, output "Unknown".

3.  **CATEGORY-SPECIFIC RULES:**
    * **For "Who":** Prefer names. If asking "Who supports him?", distinct roles like "the manager" are acceptable.
    * **For "Where":** Must be a **Named Location** (e.g., "Mars") unless the question asks for a generic type. Reject "home planet" if a specific name appears.
    * **For "What":** Extract the **Direct Object** of the action.

4.  **NOISE FILTERING:**
    * Ignore `[Image: ...]` descriptions.
    * Strip introductory modifiers (e.g., remove "a pilot named").

---
**FORMAT DEMONSTRATIONS (Out-of-Domain Examples):**

**Type 1: "Who" (Proper Name Preferred - Sci-Fi Context)**
*Input Context:* "The spaceship was piloted by Captain Nexus and a droid."
*Question:* Who piloted the ship?
*Answer:* Captain Nexus

**Type 2: "Who" (Role Fallback - Office Context)**
*Input Context:* "I submitted the report to my supervisor and the HR department." (No names given)
*Question:* Who received the report?
*Answer:* supervisor, HR department

**Type 3: "Where" (Specific vs Generic - Travel Context)**
*Input Context:* "The explorer traveled from his home country of France to the jungle."
*Question:* Where did he travel from?
*Bad Answer:* home country
*Good Answer:* France

**Type 4: "What" (Direct Object - Cooking Context)**
*Input Context:* "The chef is chopping fresh basil to make pesto."
*Question:* What is the chef chopping?
*Bad Answer:* making pesto, herbs (Goal/Generic)
*Good Answer:* fresh basil (Direct Object)

---
**Your Answer:**
"""
    return CATEGORY_C1_C4_PROMPT

def answer_c1_c4_v4(context, question):
    CATEGORY_C1_C4_PROMPT = f"""
Task: Extract **Specific Named Entities** or **Key Concepts** from the [Retrieved Memory] to answer the [Question].

[Retrieved Memory]:
{context}

[Question]: 
{question}

---
**CRITICAL EXTRACTION RULES:**

1.  **ENTITY HIERARCHY (The Most Important Rule):**
    * **For "Who" (People):** Extract **PROPER NAMES** only (e.g., "David", "Jean"). **IGNORE** generic descriptions like "a woman", "people in need", "volunteers", or visual descriptions from images.
    * **For "Where" (Places):** Extract **SPECIFIC LOCATIONS** (e.g., "Pacific Northwest", "New York"). **IGNORE** generic scenery like "park", "waterfall", "nearby area".
    * **For "What" (Things/Causes):** Extract the core subject nouns. **IGNORE** verbs or general actions.

2.  **NOISE FILTERING:**
    * **Ignore Image Descriptions:** Do not extract details that likely come from `[Image: ...]` tags (e.g., "a dog", "a woman in a mask").
    * **Strip Modifiers:** Remove "a friend named", "my colleague", "the area of". Keep only the specific name (e.g., "a colleague named Rob" -> "Rob").

3.  **FORMATTING:**
    * **No Sentences:** Start directly with the entity.
    * **Comma Separated:** If multiple specific entities exist, list them all.

---
**FORMAT DEMONSTRATIONS (Structure matched to input):**

**Type 1: Person Extraction (Who)**
*Input Context:* "(Score: 0.88) [Time: ...] [Content]: I met a lot of people at the shelter, including a tall man and my old friend Steve. [Image: a crowd of people standing near a table]"
*Question:* Who did the speaker meet?
*Bad Answer:* a tall man, people at shelter, Steve, crowd (Includes generic/visuals)
*Good Answer:* Steve

**Type 2: Location Extraction (Where)**
*Input Context:* "(Score: 0.90) [Time: ...] [Content]: We drove past some beautiful mountains and stopped in Seattle."
*Question:* What places did they visit?
*Bad Answer:* mountains, beautiful scenery, Seattle
*Good Answer:* Seattle

**Type 3: Entity Cleaning (Modifiers)**
*Input Context:* "(Score: 0.82) [Time: ...] [Content]: I went with my cousin Vinny."
*Question:* Who did they go with?
*Bad Answer:* a cousin named Vinny
*Good Answer:* Vinny

---
**Your Answer:**
"""
    return CATEGORY_C1_C4_PROMPT

def answer_c1_c4_v3(context, question):
    CATEGORY_C1_C4_PROMPT = f"""
Task: Extract specific entities or fragments from the [Context] to answer the [Question].

---
[Context]:
{context}

[Question]: 
{question}

---
**CRITICAL FORMATTING RULES:**
1. **NO SENTENCES**: Do not include "She is...", "It is...". Start directly with the answer.
2. **NO GENERALIZATION**: List specific items found in the text.
3. **EVIDENCE**: For Yes/No, add the specific reason starting with "because" or "since".

---
**FORMAT DEMONSTRATIONS (Strictly follow this style):**

**Type 1: List Extraction**
*Context:* "To make the cake, you need flour, sugar, eggs, and butter."
*Question:* What ingredients are needed?
*Bad Answer:* You need ingredients like flour and dairy. (Too general / Sentence)
*Good Answer:* flour, sugar, eggs, butter

**Type 2: Attribute Extraction**
*Context:* "The old house was painted a bright yellow color."
*Question:* What color is the house?
*Bad Answer:* The house is yellow. (Sentence)
*Good Answer:* bright yellow

**Type 3: Reasoning**
*Context:* "The match was cancelled due to heavy rain."
*Question:* Did they play the match?
*Bad Answer:* No, they didn't play. (Missing evidence)
*Good Answer:* No, due to heavy rain

---
**Your Answer:**
"""
    return CATEGORY_C1_C4_PROMPT


def answer_c1_c4_v2(context, question):
    CATEGORY_C1_C4_PROMPT = f"""
Task: Extract specific information from the text to answer the question strictly and concisely.

RULES:
1. **NO COMPLETE SENTENCES.** Do not include subjects or verbs (e.g., DELETE "She is...", "They are...", "The answer is...").
2. **LISTS:** Just list the items separated by commas (e.g., "apples, bananas").
3. **YES/NO:** Start with "Yes" or "No" or "Likely yes/no", optionally followed by a fragment phrase (e.g., "Yes, due to her training").
4. **Exact words:** Prefer exact words from the text where possible.

---------------------
EXAMPLES:

Context: "John went to the market to buy apples, pears, and some milk. He loves fresh fruit."
Question: What did John buy?
Bad Answer: John bought apples, pears, and milk.
Good Answer: apples, pears, milk

Context: "Despite the rain, Sarah decided to go hiking because she had new waterproof gear."
Question: Did Sarah go hiking?
Bad Answer: Yes, Sarah decided to go hiking because she had gear.
Good Answer: Yes, because she had new waterproof gear

Context: "The car is painted a bright, metallic blue."
Question: What color is the car?
Bad Answer: The car is metallic blue.
Good Answer: metallic blue
---------------------

Context:
{context}

Question: {question}
Answer:"""
    return CATEGORY_C1_C4_PROMPT

def answer_c1_c4_v1(context, question):
    CATEGORY_C1_C4_PROMPT = f"""Read the following context and answer the question using ONLY the exact words found in the text.
            
            Constraints:
            - Do NOT write a complete sentence.
            - Do NOT repeat the subject (e.g., do not say "Speaker A is...").
            - Do NOT strictly follow grammar if it makes the answer too long; prioritize the entity itself.
            - Just provide the label, identity, or entity.

            ---------------------
            {context}
            ---------------------

            Question: {question}
            Short Answer:"""
    return CATEGORY_C1_C4_PROMPT

def answer_c1_c4_v0(context, question):
    CATEGORY_C1_C4_PROMPT = f"""Based on the context: 
            ---------------------
            {context}
            ---------------------

            write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible. 

            Question: {question} Short answer:
            """
    return CATEGORY_C1_C4_PROMPT
