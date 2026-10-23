def analyze_content_prompt_gpt_v0(content):
    ANALYZE_CONTENT_PROMPT = f"""
You are an expert Knowledge Graph Extractor. Your task is to analyze the [TARGET TURN] to extract structured metadata.

Input Text:
{content}

---
Guidelines:

1. **Keywords (Entities & Concepts)**: 
   - **GOAL**: Extract 3-5 specific **Noun Phrases**.
   - **WHAT TO EXTRACT**:
     1. **Concrete Entities**: People, Locations, Organizations, Physical Objects (e.g., "painting", "sunset", "colors").
     2. **Key Concepts**: Important abstract themes distinct to the content (e.g., "identity", "freedom", "therapy").
   - **WHAT TO REJECT (Strict Filters)**:
     - **Generic/Vague Nouns**: REJECT "thing", "stuff", "way", "lot", "bit", "one".
     - **Verbs/Sentences**: Keywords must be Noun Phrases.
     - **Pronouns**: REJECT "it", "he", "she" unless part of a proper name.
   - **CONVERSATIONAL HANDLING**: 
     - Ignore conversational wrappers (e.g., "Wow", "Thanks", "That's great"). 
     - **EXTRACT** entities mentioned *within* the conversation. 
     - Only return [] if the text contains **NO** specific nouns (e.g., "Okay, see you later.").

2. **Context (Factual Restatement)**:
   - **GOAL**: Rewrite the text into a self-contained factual statement.
   - **CONSTRAINT 1**: Resolve "I/my/we" to the Speaker's name if available.
   - **CONSTRAINT 2**: Keep the tone objective. Remove conversational fillers ("Wow", "Haha").
   - **CONSTRAINT 3**: For "it" or "that", if the antecedent is clear in the text, replace it; otherwise use "the item".

---
Output Format (JSON Only):
{{
    "keywords": ["entity1", "entity2"], 
    "context": "Melanie thinks the painting is vivid." 
}}
"""
    return ANALYZE_CONTENT_PROMPT

def analyze_content_prompt_factdense(content):
    """Exhaustive-extraction variant for fact-dense, non-conversational chunks.

    The default v3 prompt caps keywords at 3-5 and restates the input as a single
    sentence -- correct when one note is one conversational turn, but catastrophic
    when one note is a 512-token block of ~30 independent facts (MemoryAgentBench
    FactConsolidation / RULER): ~85% of entities get no keyword anchor and the
    single-sentence context drops most facts. Here we extract EVERY entity and
    preserve EVERY fact, so keyword retrieval can reach any individual fact.
    """
    ANALYZE_CONTENT_PROMPT = f"""
You are an expert Knowledge Graph Extractor. The input is a block of many independent factual statements (not a conversation). Extract structured metadata that preserves ALL of them.

Input Text:
{content}

---
Guidelines:

1. **Keywords (Entities)**:
   - **GOAL**: Extract EVERY distinct **Noun Phrase** (every Person, Location, Organization, Work, Concept) that appears in the text. Be exhaustive -- do NOT summarize or pick only the salient ones. If there are 30 entities, return 30.
   - **CRITICAL SYNTAX FILTER**: Keywords CANNOT contain punctuation ('?', '!', ',') or verbs. Reject abstract fillers ("thing", "stuff", "way").
   - Each keyword must be a proper noun or concrete noun phrase copied from the text.

2. **Context (Factual Preservation)**:
   - **GOAL**: Restate ALL facts present in the input, one per line, faithfully. Do NOT drop, merge, or reorder facts. Preserve exact entity names and numbering if present.
   - **CONSTRAINT (Strict Fidelity)**: ONLY use information present in the Input Text. Do not add or infer.

---
Output Format (JSON Only):
{{
    "keywords": ["Entity One", "Entity Two", "Entity Three"],
    "context": "0. Fact one.\\n1. Fact two.\\n2. Fact three."
}}
"""
    return ANALYZE_CONTENT_PROMPT


def analyze_content_prompt_v3(content):
    ANALYZE_CONTENT_PROMPT = f"""
You are an expert Knowledge Graph Extractor. Your task is to analyze the [TARGET TURN] to extract structured metadata.

Input Text:
{content}

---
Guidelines:

1. **Keywords (Entities)**: 
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

2. **Context (Factual Restatement)**:
   - **GOAL**: Rewrite the text into a self-contained factual statement.
   - **CONSTRAINT 1 (Strict Fidelity)**: **ONLY use information present in the Input Text.**
   - **CONSTRAINT 2 (Safe Resolution)**: 
     - Resolve "I/my/we" using the Speaker's name **IF** it appears in the Input Text (e.g., "Speaker Melanie: ...").
     - **CRITICAL**: For external pronouns ("it", "that", "he") where the antecedent is missing, **keep the pronoun** or use a generic term like "the item" or "the person". **DO NOT GUESS.**
   - **CONSTRAINT 3 (No Meta-Language)**: Start directly with the subject. Avoid "The speaker says...".

---
Output Format (JSON Only):
{{
    "keywords": ["entity1", "entity2"], 
    "context": "Melanie thinks the item is cool." 
}}
"""
    return ANALYZE_CONTENT_PROMPT

def analyze_content_prompt_v2(content):
    ANALYZE_CONTENT_PROMPT = f"""
You are an expert Knowledge Graph Extractor. Your task is to analyze the [TARGET TURN] to extract structured metadata.

Input Text:
{content}

---
Guidelines:

1. **Keywords (Entities)**: 
   - **GOAL**: Extract 3-5 specific **Noun Phrases** explicitly present in the text.
   - **FOCUS PRIORITIES**:
     1. **Proper Nouns**: People (e.g., "Melanie"), Locations, Organizations.
     2. **Concrete Objects**: Physical items (e.g., "painting", "plate", "contract").
     3. **Specific Topics**: "LGBTQ support group", "deadline".
   - **CRITICAL STOP LIST (Filter these out)**: 
     - **IGNORE** conversational meta-roles.
     - **IGNORE** abstract terms.
     - **IGNORE** speaker names acting purely as subjects.
   - **ZERO-SHOT RULE**: If the text is purely phatic (e.g., "Wow", "That's cool", "What now?"), return an empty list []. **Do not force keywords.**

2. **Context (Factual Restatement)**:
   - **GOAL**: Rewrite the text into a self-contained factual statement.
   - **CONSTRAINT 1 (Strict Fidelity)**: **ONLY use information present in the Input Text.**
   - **CONSTRAINT 2 (Safe Resolution)**: 
     - Resolve "I/my/we" using the Speaker's name **IF** it appears in the Input Text (e.g., "Speaker Melanie: ...").
     - **CRITICAL**: For external pronouns ("it", "that", "he") where the antecedent is missing, **keep the pronoun** or use a generic term like "the item" or "the person". **DO NOT GUESS.**
   - **CONSTRAINT 3 (No Meta-Language)**: Start directly with the subject. Avoid "The speaker says...".

---
Output Format (JSON Only):
{{
    "keywords": ["entity1", "entity2"], 
    "context": "Melanie thinks the item is cool." 
}}
"""
    return ANALYZE_CONTENT_PROMPT

def analyze_content_prompt_v1(content):
    ANALYZE_CONTENT_PROMPT = f"""Analyze the following text to extract structured metadata for a knowledge graph.

Input Text:
{content}

---
Guidelines:

1. **Keywords (Entities)**: 
   - Extract specific, low-level entities, technical terms, algorithms, metrics, or named entities.
   - **CRITICAL**: Convert terms to their **canonical singular form** (e.g., "Transformers" -> "Transformer", "running" -> "run").
   - Extract 3-8 most important keywords.

2. **Tags (Taxonomy Categories)**: 
   - Assign 1-3 high-level **Category Names** (Noun Phrases) that categorize this content.
   - **Style**: Like a library classification system (e.g., "Computer Security", "Machine Learning").
   - **Constraint**: Must be broader than Keywords.

3. **Context (Contextualized Resolution)**:
   - DO NOT summarize the text if it is already clear.
   - **CRITICAL**: If the text contains pronouns (he, she, it, they) or vague references (the event, the movie), REPLACE them with specific entity names based on logical inference.
   - If the text is a dialogue turn, clarify WHO is speaking to WHOM about WHAT.
   - Example Input: "He agreed to sign it."
   - Example Output: "User A agreed to sign the contract regarding the new project."
   - Format: A single, self-contained descriptive sentence.

---
Output Format (JSON Only):
{{
    "keywords": ["canonical_term1", "canonical_term2", ...], 
    "tags": ["Category1", "Category2"],
    "context": "Describes the mechanism of [Concept]..."
}}
            """
    return ANALYZE_CONTENT_PROMPT

GRAPH_STOP_WORDS = {
    # 元数据噪音
    "speaker", "user", "participant", "assistant", "system",
    "dialogue", "conversation", "chat", "talk", "discussion",
    "response", "reply", "question", "answer", "turn",
    "text", "context", "input", "output", "guts", # 之前发现的 bad case
    
    # 你的特定数据集的人名 (如果是双人对话，建议屏蔽)
    "melanie", "caroline", 
    
    # 常见代词 (以防万一)
    "he", "she", "it", "they", "i", "you", "we", "me"
}

def analyze_content_prompt_v0(content):
    ANALYZE_CONTENT_PROMPT = f"""Analyze the following text to extract structured metadata for a knowledge graph.

Input Text:
{content}

---
Guidelines:

1. **Keywords (Entities)**: 
   - Extract specific, low-level entities, technical terms, algorithms, metrics, or named entities.
   - **CRITICAL**: Convert terms to their **canonical singular form** (e.g., "Transformers" -> "Transformer", "running" -> "run").
   - Extract 3-8 most important keywords.

2. **Tags (Taxonomy Categories)**: 
   - Assign 1-3 high-level **Category Names** (Noun Phrases) that categorize this content.
   - **Style**: Like a library classification system (e.g., "Computer Security", "Machine Learning").
   - **Constraint**: Must be broader than Keywords.

3. **Context (Summary)**: 
   - Write a concise summary (1-2 sentences) containing the dense information of the text.
   - **Focus**: If this text answers a question, what is the core insight?
   - Format: Direct factual statement.

---
Output Format (JSON Only):
{{
    "keywords": ["canonical_term1", "canonical_term2", ...], 
    "tags": ["Category1", "Category2"],
    "context": "Describes the mechanism of [Concept]..."
}}
            """
    return ANALYZE_CONTENT_PROMPT