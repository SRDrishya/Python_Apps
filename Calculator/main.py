from calculator import add, subtract, multiply, divide, divide_integer, modulus, exponent

number1 = float(input("Enter the first number: "))
number2 = float(input("Enter the second number: ")) 
operator = input("Enter the operator (+, -, *, /, //, %, **): ")
result = None
if operator == '+':
    result = add(number1, number2)
     
elif operator == '-':
    result = subtract(number1, number2)     

elif operator == '*':
    result = multiply(number1, number2)

elif operator == '/':
        result = divide(number1, number2)
       
elif operator == '//':
        result = divide_integer(number1, number2)

elif operator == '%':
    result = modulus(number1, number2)
elif operator == '**':
    result = exponent(number1, number2)
else:
    print("Invalid operator. Please use one of the following: +, -, *, /, //, %, **.")  
if result is not None:    
    print(f"The result of {number1} {operator} {number2} is: {result}")
