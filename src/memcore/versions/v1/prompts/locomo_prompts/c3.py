def answer_c3_v2(context, question):
    CATEGORY_3_PROMPT = f"""
Based on the [Retrieved Memory], provide a **concise answer** to the [Question].

**CRITICAL INSTRUCTIONS:**
1. **Inference is Required:** Do NOT say "The provided context does not specify" or "It is unknown". You MUST infer the most likely answer based on the character's personality, interests, and common sense connections (e.g., if they like classical music, infer they likely enjoy Vivaldi; if they collect children's books, infer they likely own Dr. Seuss).
2. **Format Constraints:**
   - **Strictly** limit your answer to less than 10 words.
   - Do NOT write complete sentences.
   - For Yes/No questions (e.g., "Would...", "Is..."), you MUST start with "Yes", "No", "Likely Yes", or "Likely No".
3. **Tone:** Be direct and decisive.

[Retrieved Memory]:
{context}

[Question]:
{question}

Short answer:
    """
    return CATEGORY_3_PROMPT

def answer_c3_v1(context, question):
    CATEGORY_3_PROMPT = f"""\
Based on the [Retrieved Memory] below, write an answer in the form of **a short phrase** for the following [Question], not a sentence.
The question may need you to analyze and infer the answer from the summary.
    
[Retrieved Memory]:
{context}

[Question]:
{question}

Short answer:
"""
    return CATEGORY_3_PROMPT

def answer_c3_v0(context, question):
    CATEGORY_3_PROMPT = f"""
    Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

    Question: {question} 
    Short answer:
    """
    return CATEGORY_3_PROMPT
