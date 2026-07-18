import csv
import os
from fileinput import filename

def save_all_expenses(expenses, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["amount", "category"]
        )

        writer.writeheader()

        writer.writerows(expenses)


def load_expenses(filename):
    if not os.path.exists(filename):
        return []
    with open(filename, mode="r", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader)

    
