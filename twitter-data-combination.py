import os
import glob
import pandas as pd

folder_path = "/Users/dwaste/Desktop/Undergrad-Thesis-Repo/twitter-scrape-outputs"
combined_file_path = "/Users/dwaste/Desktop/Undergrad-Thesis-Repo/transformed-data/combined-twitter-data.csv"

def combine_csv_files(folder_path, combined_file_path):
    """
    Combines all CSV files in the given folder into one new combined CSV file.
    
    Parameters:
    folder_path (str): The path to the folder containing the CSV files.
    combined_file_path (str): The path to the new combined CSV file.
    
    Returns:
    None
    """
    # Get a list of all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    # Read each CSV file into a pandas DataFrame and concatenate them
    combined_csv = pd.concat([pd.read_csv(f) for f in csv_files])

    # Write the combined DataFrame to a new CSV file
    combined_csv.to_csv(combined_file_path, index=False, encoding="utf-8")

combine_csv_files(folder_path, combined_file_path)