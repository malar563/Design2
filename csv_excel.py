import pandas as pd

def csv_to_excel(csv_file, excel_file):
    # Read the CSV file, assuming it is comma-separated
    df = pd.read_csv(csv_file)
    
    # Save to an Excel file
    df.to_excel(excel_file, index=False)
    
    print(f"Successfully converted {csv_file} to {excel_file}")

# Example usage
csv_to_excel("mega.csv", "mega.xlsx")