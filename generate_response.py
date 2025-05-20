from llama_cpp import Llama

def generate_response(query, retrieved_chunks, model_path):
    llm = Llama(model_path=model_path)
    prompt = f"""Context: {retrieved_chunks}. Answer the question: {query}"""
    output = llm(prompt, max_tokens=256, stop=["Q:", "\n"])
    return output["choices"][0]["text"]
