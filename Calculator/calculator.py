def add(a, b):
    return a + b

def subtract(a, b):
    return a - b    

def multiply(a, b):
    return a * b

def divide(a, b):
        if b == 0:
            return None
        return a / b

def modulus(a, b):
    if b == 0:
        return None
    return a % b

def exponent(a, b):
    return a ** b

def divide_integer(a, b):
    if b == 0:
        return None
    return a // b