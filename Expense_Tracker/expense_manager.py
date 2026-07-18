from filehandler import save_expenses_to_csv


expenses = []
def add_expense():
    try:
        amount = float(input("Enter the expense amount: "))
    except ValueError:
        print("Invalid amount. Please enter a numeric value.")
        return
    
    try:
        category = input("Enter the expense category: ")
        if category.strip() == "":
            raise ValueError("Category cannot be empty.")
        if category.isdigit():
            raise ValueError("Category must be a string, not a number.")
    except ValueError:
        print("Invalid category. Please enter a valid string.")
        return
    expense = {
        "amount": amount,
        "category": category
    }
    expenses.append(expense)
    save_expenses_to_csv(expense, "expenses.csv")