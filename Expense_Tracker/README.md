# Expense Tracker

A simple Python command-line app for managing personal expenses.

## Features
- Add new expenses with an amount and category.
- View all stored expenses and total spending.
- Edit or delete an existing expense entry.
- View category-wise expense summaries.
- Search expenses by category.

## How to Run
1. Open the Expense_Tracker folder.
2. Run the script:
   ```bash
   python main.py
   ```
3. Choose an option from the menu to manage expenses.

## Data Storage
- Expenses are saved in `expenses.csv` using CSV format.
- The app reloads data from the file each time you perform an action.

## Files
- `main.py` – menu-driven interface.
- `expense_manager.py` – handles core expense actions.
- `filehandler.py` – loads and saves expense data.
