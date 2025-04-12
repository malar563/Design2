import pandas as pd
import matplotlib.pyplot as plt

def plot_csv_data(csv_file1, csv_file2, common_column, y_column):
    # Read the CSV files
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)
    
    # Convert column names to strings to avoid KeyError
    df1.columns = df1.columns.astype(str)
    df2.columns = df2.columns.astype(str)
    
    # Merge the data on the common column
    merged_df = pd.merge(df1, df2, on=str(common_column))
    
    # Plot the data with smaller markers
    plt.figure(figsize=(10, 5))
    plt.plot(merged_df[str(common_column)], merged_df[str(y_column) + "_x"], 
             label=f"{y_column} (mega.csv)", marker='o', markersize=4)  # Adjust markersize
    plt.plot(merged_df[str(common_column)], merged_df[str(y_column) + "_y"], 
             label=f"{y_column} (output.csv)", marker='s', markersize=1)  # Adjust markersize
    
    plt.xlabel(str(common_column))
    plt.ylabel("Values")
    plt.title("Comparison of Column 3 from Two CSV Files")
    plt.legend()
    plt.grid()
    plt.show()

# Example usage
plot_csv_data("Test_reponse.csv", "mega.csv", "0", "3")

