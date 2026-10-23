def topic_description_v0(keywords_str):
    TOPIC_DESCRIPTION_PROMPT = f"""You are an expert taxonomist. Analyze the following keywords from a knowledge graph cluster and generate a concise **Topic Label** or **Category Name**.

Keywords: {keywords_str}

**Instructions:**
1. **Output Style**: Generate a **Standardized Noun Phrase** (e.g., "Deep Learning Optimization", NOT "Optimizing Deep Learning").
2. **Abstraction**: Summarize the high-level concept that connects these keywords.
3. **Length**: Keep it strict: **under 8 words**. No periods.

**Examples:**
- Input: "apple, pear, banana" -> Output: "Common Edible Fruits"
- Input: "TCP, IP, router, switch" -> Output: "Network Communication Protocols"
- Input: "CNN, RNN, Transformer" -> Output: "Deep Learning Architectures"

Topic Label:"""
    return TOPIC_DESCRIPTION_PROMPT