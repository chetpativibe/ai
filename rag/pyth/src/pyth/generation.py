from llama_index.llms.openai_like import OpenAILike

MODEL: str = "qwen3-coder-next"
API_BASE: str = "http://spark-8b01.local:8080/v1"
API_KEY: str = "fake"
CONTEXT_WINDOW: int = 128000
IS_CHAT_MODEL: bool = True
TEMPERATURE: float = 0.1
llm = OpenAILike(
    model=MODEL,
    api_base=API_BASE,
    api_key=API_KEY,
    context_window=CONTEXT_WINDOW,
    is_chat_model=IS_CHAT_MODEL,
    temperature=TEMPERATURE
)

def generate_answer(question: str, context: str) -> str:
    prompt = f"""
    Answer using only the provided context.

    Rules:
    1. Include only facts relevant to the question.
    2. Respect chronological order and dates.
    3. Do not infer facts unsupported by the context.
    4. If information is insufficient, say so.

    Context:
    {context}

    Question:
    {question}
    """
    return llm.complete(prompt).text