"""Script to list and load all CSV templates in data/csv_templates/ and print a sample from each."""
import os
import pandas as pd

TEMPLATE_DIR = "data/csv_templates"

def main():
    print("CSV Templates in:", TEMPLATE_DIR)
    for fname in os.listdir(TEMPLATE_DIR):
        if fname.endswith(".csv"):
            path = os.path.join(TEMPLATE_DIR, fname)
            print(f"\n--- {fname} ---")
            try:
                df = pd.read_csv(path)
                print(df.head())
            except Exception as e:
                print(f"Error loading {fname}: {e}")

if __name__ == "__main__":
    main()
