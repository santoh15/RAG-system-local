import sys
from io import StringIO
from tavily import TavilyClient

def execute_python_code(code: str):
    """
    Executes Python code locally and returns the output (STDOUT).
    Uses a unified scope to prevent NameError with imported modules.
    Args:
        -code (str): The Python code to execute.
    Returns:
        -str: The output from the executed code or any error message.
    """
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    
    try:
        exec_scope = {}
        exec(code, exec_scope, exec_scope)
        
        output = redirected_output.getvalue()
        return output if output else "Execution successful. No terminal output."
    except Exception as e:
        return f"{type(e).__name__}: {str(e)}"
    finally:
        sys.stdout = old_stdout



def web_search(query: str):
    """
    Searches the internet for information not found in local documents.
    Args:
        -query (str): The search query.
    Returns:
        -str: A summary of the search results or an error message if the search fails.
    """
    tavily = TavilyClient(api_key="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")
    try:
        search_result = tavily.search(query, search_depth="advanced")

        context = ""
        for res in search_result['results']:
            context += f"\nSource: {res['url']}\nContent: {res['content']}\n"
        return context
    except Exception as e:
        return f"Search error: {str(e)}"