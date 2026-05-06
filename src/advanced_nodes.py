import requests
import json

def route_query(query, API_URL):
    """
    Adaptive RAG: chose the best tool (math, web, rag) based on the query type using a local LLM for routing decisions.
    """
    system_prompt = """You are an expert router for a Physics and Mathematics Assistant. Route the user query to the correct tool:
    
    - 'rag': DEFAULT ROUTE. Use this for all questions regarding physics concepts, theoretical definitions, and academic notes (Examples: "What is the Berggren basis?", "Explain complex wave packets", "How does the Gamow Shell Model work?").
    - 'math': ONLY use this if the user EXPLICITLY asks to calculate something, plot a graph, or numerically solve an equation with code (Examples: "Plot this wave function", "Calculate the integral of...").
    - 'web': ONLY use this for current events, news, or real-time internet data.
    
    Respond ONLY with the exact word: rag, math, or web. Do not add any other words or punctuation."""
    
    payload = {
        "model": "local-model", 
        "messages": [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": query}
        ], 
        "temperature": 0.0, 
        "max_tokens": 10
    }
    
    try:
        import requests
        response = requests.post(API_URL, json=payload, timeout=30)
        route_decision = response.json()['choices'][0]['message']['content'].strip().lower()
        
        if 'math' in route_decision:
            return 'math'
        elif 'web' in route_decision:
            return 'web'
        else:
            return 'rag'
    except:
        return 'rag'

def rewrite_query(query, API_URL):
    """
    Query Rewriting: Improve the query for vector database search.
    Args:
        - query (str): The user's query.
        - API_URL (str): The URL of the local LLM API for rewriting.
    Returns:
        - str: The rewritten query.
    """
    system_prompt = """You are an expert in Physics and Math. Rewrite the user's query to make it optimal for a vector database search. Extract the core scientific concepts and keywords. 
    Respond ONLY with the rewritten query, nothing else."""
    
    payload = {"model": "local-model", "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}], "temperature": 0.1, "max_tokens": 50}
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        return response.json()['choices'][0]['message']['content'].strip()
    except:
        return query

def grade_hallucination(context, answer, API_URL):
    """
    Self-RAG: Verify if the LLM invented information.
    Args:
        - context (str): The retrieved context.
        - answer (str): The LLM's answer.
        - API_URL (str): The URL of the local LLM API for grading.
    Returns:
        - str: 'yes' if the answer is grounded in the context, 'no' otherwise.
    """
    system_prompt = """You are a grader assessing whether an LLM generation is grounded in the retrieved context.
    Respond ONLY 'yes' if the answer is grounded in the context, or 'no' if it contains hallucinations or fabricated information."""
    user_content = f"CONTEXT: {context}\n\nANSWER: {answer}"
    
    payload = {"model": "local-model", "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}], "temperature": 0.0, "max_tokens": 5}
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        return 'yes' if 'yes' in response.json()['choices'][0]['message']['content'].lower() else 'no'
    except:
        return 'yes'


def get_standalone_question(chat_history, latest_query, API_URL):
    """
    Analize the last 5 messages of the chat history to rewrite the latest user query into a standalone question,
    resolving pronouns and omitted subjects. Use a local LLM for this task. If the question is already standalone, return it unchanged.
    Args:
            - chat_history (list): The list of previous messages in the conversation.
            - latest_query (str): The latest user query that may contain pronouns or omitted subjects.
            - API_URL (str): The URL of the local LLM API for rewriting the question.
    Returns:
            - str: The rewritten standalone question.
    """
    context_window = chat_history[-5:] 
    
    history_str = ""
    for msg in context_window:
        if msg['role'] == 'system': continue
        role = "Usuario" if msg['role'] == 'user' else "Asistente"
        content = msg['content'].split("[SYSTEM NOTE]")[0].strip()
        history_str += f"{role}: {content}\n"

    system_prompt = """Tu tarea es reformular la PREGUNTA ACTUAL del usuario para que sea una 
pregunta independiente (standalone). Usa el HISTORIAL para entender a qué se refieren los 
pronombres (él, eso, aquello, su) o sujetos omitidos.

REGLAS:
1. NO respondas la pregunta.
2. NO añadidas introducciones como "La pregunta reformulada es...".
3. Si la pregunta ya se entiende perfectamente sola, devuélvela idéntica.
4. Responde ÚNICAMENTE con la pregunta reformulada."""
    
    user_content = f"HISTORIAL:\n{history_str}\n\nPREGUNTA ACTUAL: {latest_query}"
    
    payload = {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.0,
        "max_tokens": 150
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        return response.json()['choices'][0]['message']['content'].strip()
    except:
        return latest_query 