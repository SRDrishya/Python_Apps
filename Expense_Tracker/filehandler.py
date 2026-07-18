import csv

def save_expenses_to_csv(expense, filename):
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["amount", "category"])
        if file.tell() == 0:  # Check if the file is empty
            writer.writeheader()
        writer.writerow(expense)

def view_expenses(filename):
    try:
        with open(filename, mode='r') as file:
            reader = csv.DictReader(file)
            expenses = list(reader)
            if not expenses:
                print("No expenses recorded.")
                return
            total = 0
            print("Expenses")
            print("-"*30)
            for expense in expenses:
                print(f"Amount: {expense['amount']}, Category: {expense['category']}")
                total += float(expense['amount'])
            print("-"*30)
            print(f"Total Expenses: {total}")
    except FileNotFoundError:
        print("No expenses recorded yet. Please add an expense first.")


def expense_summary_by_category(filename):
    try:
        with open(filename, mode='r') as file:
            reader = csv.DictReader(file)
            expenses = list(reader)
            if not expenses:
                print("No expenses recorded.")
                return
            category_summary = {}
            for expense in expenses:
                category = expense['category']
                amount = float(expense['amount'])
                if category in category_summary:
                    category_summary[category] += amount
                else:
                    category_summary[category] = amount
            print("Expense Summary by Category")
            print("-"*30)
            for category, total in category_summary.items():
                print(f"{category:<10}  Rs{total}")
    except FileNotFoundError:
        print("No expenses recorded yet. Please add an expense first.")