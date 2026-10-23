def reflector_prompt_v0(question, context):
    TOG3_REFLECTOR_PROMPT = f"""You are a reflector agent that evaluates whether the current context and current answer is sufficient to answer a question.

    Question: {question}

    Current Context and Current Answer:
    {context}

    Evaluate whether the provided context and answer contains enough information to answer the question comprehensively.

    If the context and answer is insufficient, identify what specific information is missing.

    Return your assessment as a JSON object:
    {{
        "sufficient": true or false,
        "missing_info": "detailed description of what information is missing, or empty string if sufficient",
        "confidence": 0.0 to 1.0
    }}
    """
    return TOG3_REFLECTOR_PROMPT