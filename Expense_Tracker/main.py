from expense_manager import add_expense, delete_expense, edit_expense,view_expenses ,expense_summary, search_expense


def main():
    while True:
        print("1. Add Expense")
        print("2. View Expenses")
        print("3. Edit Expense")
        print("4. Delete Expense")
        print("5. Expense Summary")
        print("6. Search Expenses")
        print("7. Exit")

        choice = input("Please select an option: ")
        
        if choice == "1":
            add_expense()

        elif choice == "2":
            view_expenses()

        elif choice == "3":
            edit_expense()

        elif choice == "4":
            delete_expense()

        elif choice == "5":
            expense_summary()

        elif choice == "6":
            search_expense()

        elif choice == "7":
            break

if __name__ == "__main__":
        main()