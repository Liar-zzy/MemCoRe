def query_intent_prompt_gpt_v0(query):
    QUERY_PROMPT = f"""Analyze the user query to identify its **Target Taxonomy Category** and extract key search entities.

**Task 1: topic_desc (Target Category)**
- Predict the **Taxonomy Category** or **Subject Heading**.
- **Style**: Strict **Noun Phrase** (e.g., "Digital Art", "Emotional State", "Travel Plans").
- **Constraint**: Keep it **under 6 words**. Avoid verbs.

**Task 2: Keywords**
- **GOAL**: Extract 3-5 specific **Noun Phrases** to be used as search queries.
- **WHAT TO EXTRACT**:
  1. **Entities**: People, Locations, Organizations, Objects (e.g., "Caroline", "sunset", "blue paint").
  2. **Key Concepts**: Specific abstract themes (e.g., "inspiration", "freedom", "marriage").
- **WHAT TO REJECT (Strict Filters)**:
  - **Generic/Vague Nouns**: REJECT "thing", "stuff", "way", "something", "anything".
  - **Verbs/Sentences**: Keywords MUST be Noun Phrases.
- **CONVERSATIONAL HANDLING (Crucial)**:
  - **Ignore wrappers**: If user asks "Do you remember X?" or "Tell me about Y", **EXTRACT X and Y**.
  - **Do NOT return empty** just because the query is polite or conversational.
  - **ONLY return []** if the query is TRULY devoid of content (e.g., "Hello there", "Are you ready?").

**Query:** {query}

**Return Format (JSON):**
{{
  "topic_desc": "Concise Noun Phrase",
  "keywords": ["keyword1", "keyword2"]
}}

Now analyze the query above."""
    return QUERY_PROMPT

def query_intent_v3(query):
	QUERY_PROMPT = f"""Analyze the user query to identify its **Target Taxonomy Category**, extract key entities, and detect time-related intent.

**Task 1: topic_desc (Target Category)**
- Predict the **Taxonomy Category** or **Subject Heading** this query falls under.
- **Style**: Strict **Noun Phrase** (like a book chapter title or library category).
- **Constraint**: Keep it **under 8 words**.
- **Do NOT** describe the user's intent (e.g., avoid "how to...", "techniques for...", "process of..."). Instead, name the *topic itself*.

**Task 2: Keywords**
- **GOAL**: Extract 3-5 specific **Noun Phrases** (Entities, Objects, Concepts).
- **CRITICAL SYNTAX FILTER (Must Validate)**:
  - **REJECT SENTENCES**: Keywords CANNOT contain punctuation like '?', '!', or ','.
  - **REJECT VERBS**: Keywords CANNOT contain verbs.
  - **REJECT ABSTRACT**: Reject "thing", "stuff", "lot", "way".
- **FOCUS PRIORITIES**:
  1. **Proper Nouns**: People (EXCLUDING the speaker), Locations, Organizations.
  2. **Concrete Objects**.
  3. **Specific Topics**.
- **ZERO-SHOT RULE**: If the text is purely conversational/phatic (e.g., "Wow", "That's cool", "What now?"), return an empty list []. **Do not force keywords.**

**Query:** {query}

**Return Format (JSON):**
{{
  "topic_desc": "Concise Noun Phrase (Category Name)",
  "keywords": ["keyword1", "keyword2", "keyword3"]
}}

Now analyze the query above."""
	return QUERY_PROMPT

def query_intent_v2(query):
	QUERY_PROMPT = f"""Analyze the user query to identify its **Target Taxonomy Category**, extract key entities, and detect time-related intent.

**Task 1: topic_desc (Target Category)**
- Predict the **Taxonomy Category** or **Subject Heading** this query falls under.
- **Style**: Strict **Noun Phrase** (like a book chapter title or library category).
- **Constraint**: Keep it **under 8 words**.
- **Do NOT** describe the user's intent (e.g., avoid "how to...", "techniques for...", "process of..."). Instead, name the *topic itself*.

**Task 2: Keywords**
- Extract 3-5 core entities, technical terms, or specific concepts.
- **CRITICAL**: Convert terms to their **canonical singular form** (e.g., "transformers" -> "Transformer", "running" -> "run").
- Exclude generic verbs or stop words. 

**Task 3: Time Sorting Constraint**
- Determine if the user **explicitly** wants the results sorted by time (e.g., looking for the "newest" or "oldest" items).
- **Rule for "latest"**: Set to "latest" ONLY IF the query contains explicit recency markers like: "recently", "latest", "last", "newest", "current", "just", "new".
- **Rule for "oldest"**: Set to "oldest" ONLY IF the query contains explicit start markers like: "first", "oldest", "earliest", "beginning", "started", "original".
- **Rule for "none" (CRITICAL)**: If the query starts with "When..." or asks for a specific date BUT does NOT contain the markers above, set to "none". (e.g., "When did he go?" is "none", but "When did he last go?" is "latest").

**Query:** {query}

**Return Format (JSON):**
{{
  "topic_desc": "Concise Noun Phrase (Category Name)",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "time_sorting": "latest" | "oldest" | "none"
}}

Now analyze the query above."""
	return QUERY_PROMPT

def query_intent_v1(query):
	QUERY_PROMPT = f"""Analyze the user query to identify its **Target Taxonomy Category**, extract key entities, and detect time-related intent.

**Task 1: topic_desc (Target Category)**
- Predict the **Taxonomy Category** or **Subject Heading** this query falls under.
- **Style**: Strict **Noun Phrase** (like a book chapter title or library category).
- **Constraint**: Keep it **under 8 words**.
- **Do NOT** describe the user's intent (e.g., avoid "how to...", "techniques for...", "process of..."). Instead, name the *topic itself*.

**Task 2: Keywords**
- Extract 3-5 core **Noun Phrases** (Entities, Technical Terms).
- **CONSTRAINT**: The knowledge graph strictly filters out verbs. **You must also REJECT verbs.**
  - Bad: ["running", "process", "make"]
  - Good: ["run cycle", "processing unit", "manufacturing"] (If they are specific nouns)
- **Canonical Form**: Convert to singular (e.g., "cats" -> "cat").

**Task 3: Time Sorting Constraint**
- Determine if the user **explicitly** wants the results sorted by time (e.g., looking for the "newest" or "oldest" items).
- **Rule for "latest"**: Set to "latest" ONLY IF the query contains explicit recency markers like: "recently", "latest", "last", "newest", "current", "just", "new".
- **Rule for "oldest"**: Set to "oldest" ONLY IF the query contains explicit start markers like: "first", "oldest", "earliest", "beginning", "started", "original".
- **Rule for "none" (CRITICAL)**: If the query starts with "When..." or asks for a specific date BUT does NOT contain the markers above, set to "none". (e.g., "When did he go?" is "none", but "When did he last go?" is "latest").

**Query:** {query}

**Return Format (JSON):**
{{
  "topic_desc": "Concise Noun Phrase (Category Name)",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "time_sorting": "latest" | "oldest" | "none"
}}

**Comparison Examples:**

Query: "How do neural networks process images?"
topic_desc: "Computer Vision and Deep Learning"
keywords: ["neural network", "image", "processing"]
time_sorting: "none"

Query: "What did John recently finish reading?"
topic_desc: "Personal Reading Activity"
keywords: ["John", "reading", "book"]
time_sorting: "latest"

Query: "What was the first project Alice worked on?"
topic_desc: "Personal Work History"
keywords: ["Alice", "project", "work"]
time_sorting: "oldest"

Query: "What are the latest developments in AI?"
topic_desc: "Artificial Intelligence Advances"
keywords: ["AI", "development", "progress"]
time_sorting: "latest"

Now analyze the query above."""
	return QUERY_PROMPT

def query_intent_v0(query):
	QUERY_PROMPT = f"""Analyze the user query to identify its **Target Taxonomy Category**, extract key entities, and detect time-related intent.

**Task 1: topic_desc (Target Category)**
- Predict the **Taxonomy Category** or **Subject Heading** this query falls under.
- **Style**: Strict **Noun Phrase** (like a book chapter title or library category).
- **Constraint**: Keep it **under 8 words**.
- **Do NOT** describe the user's intent (e.g., avoid "how to...", "techniques for...", "process of..."). Instead, name the *topic itself*.

**Task 2: Keywords**
- Extract 3-5 core entities, technical terms, or specific concepts.
- **CRITICAL**: Convert terms to their **canonical singular form** (e.g., "transformers" -> "Transformer", "running" -> "run").
- Exclude generic verbs or stop words.

**Task 3: Time Sorting Constraint**
- Determine if the user **explicitly** wants the results sorted by time (e.g., looking for the "newest" or "oldest" items).
- **Rule for "latest"**: Set to "latest" ONLY IF the query contains explicit recency markers like: "recently", "latest", "last", "newest", "current", "just", "new".
- **Rule for "oldest"**: Set to "oldest" ONLY IF the query contains explicit start markers like: "first", "oldest", "earliest", "beginning", "started", "original".
- **Rule for "none" (CRITICAL)**: If the query starts with "When..." or asks for a specific date BUT does NOT contain the markers above, set to "none". (e.g., "When did he go?" is "none", but "When did he last go?" is "latest").

**Query:** {query}

**Return Format (JSON):**
{{
  "topic_desc": "Concise Noun Phrase (Category Name)",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "time_sorting": "latest" | "oldest" | "none"
}}

**Comparison Examples:**

Query: "How do neural networks process images?"
topic_desc: "Computer Vision and Deep Learning"
keywords: ["neural network", "image", "processing"]
time_sorting: "none"

Query: "What did John recently finish reading?"
topic_desc: "Personal Reading Activity"
keywords: ["John", "reading", "book"]
time_sorting: "latest"

Query: "What was the first project Alice worked on?"
topic_desc: "Personal Work History"
keywords: ["Alice", "project", "work"]
time_sorting: "oldest"

Query: "What are the latest developments in AI?"
topic_desc: "Artificial Intelligence Advances"
keywords: ["AI", "development", "progress"]
time_sorting: "latest"

Now analyze the query above."""
	return QUERY_PROMPT