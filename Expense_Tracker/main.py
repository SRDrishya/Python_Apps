from expense_manager import add_expense
from filehandler import expense_summary_by_category, view_expenses


def main():
    while True:
        print("Welcome to the Expense Tracker!")
        print("1. Add Expense")
        print("2. View Expenses")
        print("3. View Expense Summary by Category")
        print("4. Exit")

        choice = input("Please select an option: ")
        
        if choice == '1':
            add_expense()
        elif choice == '2':
            view_expenses("expenses.csv")
        elif choice == '3':
           expense_summary_by_category("expenses.csv")
        elif choice == '4':
            print("Exiting the Expense Tracker. Goodbye!")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()