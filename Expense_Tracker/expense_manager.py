from filehandler import (
    load_expenses,
    save_all_expenses
)
FILENAME = "expenses.csv"
expenses = []
def add_expense():

    expenses = load_expenses(FILENAME)

    try:
        amount = float(input("Enter amount: "))

        if amount <= 0:
            print("Amount must be greater than 0.")
            return

    except ValueError:
        print("Invalid amount.")
        return

    category = input("Enter category: ").strip()

    if not category:
        print("Category cannot be empty.")
        return

    if category.isdigit():
        print("Category cannot be only numbers.")
        return

    expense = {
        "amount": amount,
        "category": category
    }

    expenses.append(expense)

    save_all_expenses(expenses, FILENAME)

    print("Expense added successfully.")

def delete_expense():

    expenses = load_expenses(FILENAME)

    if not expenses:
        print("No expenses found.")
        return

    for index, expense in enumerate(expenses, start=1):
        print(f"{index}. ₹{expense['amount']} - {expense['category']}")

    try:
        choice = int(input("Enter expense number to delete: "))

        if choice < 1 or choice > len(expenses):
            print("Invalid choice.")
            return

        deleted = expenses.pop(choice - 1)

        save_all_expenses(expenses, FILENAME)

        print(f"Deleted: ₹{deleted['amount']} - {deleted['category']}")

    except ValueError:
        print("Please enter a valid number.")

def edit_expense():

    expenses = load_expenses(FILENAME)

    if not expenses:
        print("No expenses found.")
        return

    for index, expense in enumerate(expenses, start=1):
        print(f"{index}. ₹{expense['amount']} - {expense['category']}")

    try:
        choice = int(input("Select expense to edit: "))

        if choice < 1 or choice > len(expenses):
            print("Invalid selection.")
            return

        amount = float(input("Enter new amount: "))

        if amount <= 0:
            print("Amount must be greater than 0.")
            return

        category = input("Enter new category: ").strip()

        if not category:
            print("Category cannot be empty.")
            return

        expenses[choice - 1]["amount"] = amount
        expenses[choice - 1]["category"] = category

        save_all_expenses(expenses, FILENAME)

        print("Expense updated successfully.")

    except ValueError:
        print("Invalid input.")

def view_expenses():

    expenses = load_expenses(FILENAME)

    if not expenses:
        print("No expenses found.")
        return

    print("\nExpense List")
    print("-" * 35)

    total = 0

    for expense in expenses:

        print(
            f"₹{expense['amount']}  |  {expense['category']}"
        )

        total += float(expense["amount"])

    print("-" * 35)
    print(f"Total: ₹{total}")

def expense_summary():

    expenses = load_expenses(FILENAME)

    if not expenses:
        print("No expenses found.")
        return

    summary = {}

    for expense in expenses:

        category = expense["category"]
        amount = float(expense["amount"])

        if category in summary:
            summary[category] += amount
        else:
            summary[category] = amount

    print("\nExpense Summary")
    print("-" * 35)

    for category, total in summary.items():
        print(f"{category:<15} ₹{total}")

def search_expense():

    expenses = load_expenses(FILENAME)

    if not expenses:
        print("No expenses found.")
        return

    category = input("Enter category to search: ").strip()

    found = False

    print("\nMatching Expenses")
    print("-" * 30)

    for expense in expenses:

        if expense["category"].lower() == category.lower():

            print(
                f"₹{expense['amount']} | {expense['category']}"
            )

            found = True

    if not found:
        print("No matching expenses found.")