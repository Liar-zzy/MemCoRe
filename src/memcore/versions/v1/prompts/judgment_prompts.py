def judgment_prompts_v4_gating(content, context, keywords, candidates_str):
    JUDEMENT_PROMPT = f"""
### Role
You are a Knowledge Graph Updater. Your job is to link a NEW NODE to existing CANDIDATE NODES.

### Input Data
[NEW NODE]
Content: "{content}"
Context: "{context}"
Keywords: {keywords}

[CANDIDATE NODES]
{candidates_str}

### Instructions
Analyze each candidate and generate a JSON response. Follow these STRICT rules for updating context:

1. **Analyze Link**:
   - Determine `relation_type`: 'SUPPORTS', 'CONFLICTS', or 'RELATED_TO'.
   - Assign `connection_strength` (0.0 - 1.0).

2. **Update Context (Apply Rules based on Strength)**:
   - **CASE A: Strength >= 0.8 (Strong)** -> **MERGE DETAILS**.
     * Action: Add specific details from the New Node to the candidate's context.
     * Template: "[Original Context]. Specifically, [New Node Info]..."
   
   - **CASE B: Strength < 0.5 (Weak)** -> **ADD TAG ONLY**.
     * Action: Keep original context exactly as is. Just append a reference tag.
     * Template: "[Original Context]. (Related: [New Node Keyword])"
   
   - **CASE C: relation_type is 'CONFLICTS'** -> **ADD CONTRAST**.
     * Action: Note the difference.
     * Template: "[Original Context]. However, [New Node] argues that..."

3. **Output**: Return strictly valid JSON matching the schema.
"""
    return JUDEMENT_PROMPT

def judgment_prompts_v3_gating(content, context, keywords, candidates_str):
    JUDEMENT_PROMPT = f"""You are the MemDyn Evolution Engine. Integrate the NEW MEMORY into the graph with SEMANTIC GATING.

[NEW MEMORY]
Content: "{content}"
Context: "{context}"
Keywords: {keywords}

[CANDIDATES]
{candidates_str}

[TASK]
1. **Link Analysis**: For each candidate, determine:
   - `relation_type`: 'SUPPORTS', 'CONFLICTS', or 'RELATED_TO'.
   - `connection_strength` (0.0-1.0): 
     - 1.0: Crucial/Direct relation. 
     - 0.5: Moderate relevance.
     - 0.1: Weak relevance.

2. **Context Refinement (Semantic Gating)**: Update neighbors based on their strength:
   - **Strength > 0.8 (Deep Integration)**: Rewrite context to deeply weave in the new memory.
   - **Strength < 0.5 (Peripheral)**: Only add a minor tag/reference. Keep original context intact.
   - **CONFLICTS**: Update context to explicitly state the difference/contradiction.

**CRITICAL RULES FOR `new_context`**:
1. **Omniscient Tone**: Write strictly as a factual statement about the world/person.
2. **Preservation**: You MUST retain the candidate's original core facts. Do not lose old info.
3. **Integration**: Weave the new details into the existing sentence structure if possible.

**Return JSON:**
{{
    "neighbor_updates": [
        {{
            "id": "candidate_id",
            "new_context": "The refined context based on gating strength.",
            "rationale": "Brief explanation of gating decision."
        }}
    ],
    "new_links": [
        {{
            "target_id": "candidate_id",
            "relation_type": "SUPPORTS",
            "connection_strength": 0.85,
            "reason": "Brief explanation of the connection."
        }}
    ]
}}
"""
    return JUDEMENT_PROMPT

def judgment_prompts_v2(content, context, keywords, candidates_str):
    JUDEMENT_PROMPT = f"""You are an Omniscient Memory Integrator.
You are processing a [New Memory] and integrating its details into [Existing Memories] to build a coherent knowledge graph.

[New Memory]:
- Content: {content}
- Context: {context}
- Keywords: {keywords}

[Candidates] (Existing memories):
{candidates_str}

---

**Task 1: Evolution (Merge & Refine Context)**
Check if the [New Memory] adds valuable details (who, what, where, why) to a [Candidate].
If yes, REWRITE the [Candidate]'s `context` to seamlessly integrate the new details.

**CRITICAL RULES FOR `new_context`**:
1.  **Omniscient Tone**: Write strictly as a factual statement about the world/person.
    -   *YES*: "Melanie runs to de-stress and recently participated in a charity race."
    -   *NO*: "The new memory indicates Melanie runs." (No meta-language!)
    -   *NO*: "Speaker Melanie said..." (No dialogue labels!)
2.  **Preservation**: You MUST retain the [Candidate]'s original core facts. Do not lose old info.
3.  **Integration**: Do not just append sentences. Wove the new details into the existing sentence structure if possible.
4.  **Banned Phrases**: NEVER use words like "indicates", "suggests", "mentions", "updates", "refers to", "according to".

**Task 2: Linking (Structure)**
Check for HARD LOGICAL CONNECTIONS based on the `match_reason` and content.
-   **Shared Keywords**: If `match_reason` shows shared entities (e.g., "Shared Keywords [Melanie, Painting]"), this is a strong signal for a LINK.
-   **Logic**: Causality, Sequence, Identity (talking about the same specific object/event).
-   **Constraint**: ONLY link if `constraints.allow_link` is true (or if the candidate is marked as a linking candidate).

---

**Example of Evolution:**
* **Candidate Context**: "Melanie enjoys painting landscapes."
* **New Memory**: "I painted a lake sunrise yesterday."
* **BAD Update**: "The new memory mentions she painted a lake sunrise." (Meta-language failure)
* **BAD Update**: "Melanie painted a lake sunrise." (Lost the "enjoys painting landscapes" part)
* **GOOD Update**: "Melanie enjoys painting landscapes, specifically creating a lake sunrise piece recently." (Perfect integration)

---

**Return JSON:**
{{
    "neighbor_updates": [
        {{
            "id": "candidate_id",
            "new_context": "The unified, omniscient factual statement."
        }}
    ],
    "new_links": [
        "candidate_id" 
    ]
}}
"""
    return JUDEMENT_PROMPT

def judgment_prompts_v1(content, context, keywords, candidates_str):
    JUDEMENT_PROMPT = f"""You are a Knowledge Graph Curator. You are adding a [New Memory] and checking if it affects [Existing Memories].

[New Memory]:
- Content: {content}
- Context: {context}
- Keywords: {keywords}

[Candidates] (Existing memories with match reasons):
{candidates_str}

---

**Understanding match_reason:**
Each candidate has a `match_reason` field explaining WHY it was retrieved:
- "High Semantic Similarity (score=0.XX)" → The texts are semantically similar (embedding-based).
- "Shared Keywords [keyword1, keyword2, ...]" → These specific entities/concepts appear in BOTH memories.

**IMPORTANT**: Pay special attention to "Shared Keywords" matches! Even if the content looks different on the surface, shared keywords indicate they are discussing the SAME entities (e.g., same person, same event, same object). These are often excellent candidates for LINKING.

---

Your Task:
Evaluate if the [New Memory] provides specific details that make a [Candidate] more complete or accurate.

1. **Evolution (Update Context)**:
   - Does [New Memory] provide missing details (names, dates, reasons) for [Candidate]?
   - If YES, write a `new_context` for the [Candidate].
   - **CRITICAL RULE**: The `new_context` must describe the **[Candidate]**, NOT the [New Memory]. It must be the [Candidate]'s original context enriched with the new details.
   - *Bad Example*: "New Memory mentions a painting." (This just describes the source)
   - *Good Example*: "User A likes the [Lake Sunrise] painting (previously just 'a painting'), which connects to their hobby of art." (This merges old and new info)

2. **Linking (Structure)**:
   - Is there a HARD LOGICAL CONNECTION? (Causality, Sequence, Explicit Reference, **Same Entity**).
   - **HINT**: If `match_reason` shows "Shared Keywords", there's likely a connection worth establishing!
   - Only link if strict logic applies.

---

**Return JSON:**
{{
    "neighbor_updates": [
        {{
            "id": "candidate_id",
            "new_context": "The MERGED context: The original context + specific details from New Memory."
        }}
    ],
    "new_links": [
        "candidate_id" 
    ]
}}

**Guidelines:**
- **Do NOT overwrite**: Do not replace the candidate's context with a summary of the New Memory. **Augment it.**
- **Be Conservative**: If the New Memory is just a duplicate or general comment, do not update.
- **Trust Shared Keywords**: If two memories share specific keywords, they are likely related even if the wording differs.
"""
    return JUDEMENT_PROMPT

def judgment_prompts_v0(content, context, keywords, candidates_str):
    JUDEMENT_PROMPT = f"""You are a Memory Evolution Agent. A new memory is being added to a knowledge graph.

Your task is TWOFOLD:

1. **Evolution (Update Neighbors)**: Does this new memory clarify, correct, or add valuable context to existing memories?
   - If yes, generate updated 'context' (1-2 sentences) for those memories.
   - Only update if the new memory genuinely enhances understanding.

2. **Linking (Structure)**: Is there a HARD LOGICAL CONNECTION to existing memories?
   - Types: Causality (causes/explains), Sequence (follows/precedes), Reference (directly mentions), Contradiction (conflicts with)
   - ONLY link if one of these STRICT relationships exists.
   - High similarity alone is NOT enough for linking.

---

**New Memory:**
- Content: {content}
- Context: {context}
- Keywords: {keywords}

**Candidates (sorted by similarity):**
{candidates_str}

---

**Return JSON:**
{{
    "neighbor_updates": [
        {{
            "id": "existing_note_id",
            "new_context": "Updated context (1-2 sentences, be specific about what this memory discusses)",
        }}
        // Only include memories that genuinely benefit from the new information
    ],
    "new_links": [
        "existing_note_id"  // ONLY if STRICT logical connection exists (candidates with is_linking_candidate=true)
    ]
}}

**Guidelines:**
- Be conservative: empty arrays are valid if no updates/links are appropriate.
- For updates: focus on adding NEW insights, not rephrasing existing context.
- For links: require explicit logical relationships, not just topical similarity.
"""
    return JUDEMENT_PROMPT


def judgment_prompts_v3_gating_corrected(content, context, keywords, candidates_str):
    """E3: Corrected v3_gating prompt — removed in-prompt threshold numbers.
    All merge/link/append decisions are now made by code-side thresholds (τ_m=0.7, τ_l=0.5).
    """
    JUDEMENT_PROMPT = f"""You are the MemDyn Evolution Engine. Integrate the NEW MEMORY into the graph with SEMANTIC GATING.

[NEW MEMORY]
Content: "{content}"
Context: "{context}"
Keywords: {keywords}

[CANDIDATES]
{candidates_str}

[TASK]
1. **Link Analysis**: For each candidate, determine:
   - `relation_type`: 'SUPPORTS', 'CONFLICTS', or 'RELATED_TO'.
   - `connection_strength` (0.0-1.0):
     - 1.0: Crucial/Direct relation.
     - 0.5: Moderate relevance.
     - 0.1: Weak relevance.
   - Do NOT make merge/link/append decisions. Only output the relation analysis.

2. **Context Refinement**: If `relation_type` is 'SUPPORTS' and strength is high, provide an updated context that integrates both nodes' information.
   - For `CONFLICTS`: Update context to explicitly state the difference/contradiction.

**CRITICAL RULES FOR `new_context`**:
1. **Omniscient Tone**: Write strictly as a factual statement about the world/person.
2. **Preservation**: You MUST retain the candidate's original core facts. Do not lose old info.
3. **Integration**: Weave the new details into the existing sentence structure if possible.

**Return JSON:**
{{
    "neighbor_updates": [
        {{
            "id": "candidate_id",
            "new_context": "The refined context integrating new information.",
            "rationale": "Brief explanation of the integration decision."
        }}
    ],
    "new_links": [
        {{
            "target_id": "candidate_id",
            "relation_type": "SUPPORTS",
            "connection_strength": 0.85,
            "reason": "Brief explanation of the connection."
        }}
    ]
}}
"""
    return JUDEMENT_PROMPT
