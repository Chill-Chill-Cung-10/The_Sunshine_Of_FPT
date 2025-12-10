from langchain_core.tools import tool

@tool("sum")
def sum(a, b):
    """
    calculator sum of 2 number 
    
    :param a: number 1
    :param b: number 2
    """
    return int(a) + int(b)