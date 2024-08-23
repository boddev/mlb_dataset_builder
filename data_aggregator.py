import os
from datetime import datetime, timedelta
import pandas as pd

# Define the start and end dates for the search
start_date = datetime(2023, 3, 30)
end_date = datetime(2023, 10, 31)

# Define the time delta for the search (7 days)
delta = timedelta(days=7)

# Create an empty list to store the files within 7 days of each date
files_within_7_days = []

# Loop through each date between the start and end dates
for date in range(0,(end_date - start_date).days,8):
    # Calculate the current date
    current_date = start_date + timedelta(date)
    
    # Loop through each file in the 'files' directory
    for file in os.listdir('files'):
        # Check if the file is a CSV file
        if file.endswith('.csv'):
            # Extract the date from the file name
            file_date = datetime.strptime(file[:8], '%Y%m%d')
            
            # Check if the file date is within 7 days of the current date
            if abs(current_date - file_date) <= delta:
                # Add the file to the list of files within 7 days of the current date
                files_within_7_days.append(file)
            else:
                #process and aggregate data here
                combined_results = pd.DataFrame()

                # Loop through each file in files_within_7_days
                
                for file in files_within_7_days:
                    col_name = 'batting'
                    if file.find('pitching') > -1:
                        continue
                        #col_name = 'pitching'
                    # Read the contents of the file into a DataFrame
                    df = pd.read_csv(os.path.join('files', file))
                    
                    for r, row in df.iterrows():
                        name = ''
                        if col_name == "batting":
                            theName = row[col_name.capitalize()].split()
                            for part in range(len(theName)-1):
                                name = name + ' '+theName[part]
                        else:
                            theName = row[col_name.capitalize()].split(',')
                            name = theName[0]
                        
                        df.loc[r, col_name.capitalize()] = name.strip()
                    
                    combined_results = pd.concat([df,combined_results])
                # Drop duplicates from the combined results DataFrame
                drop_dups = combined_results.drop_duplicates(subset=[col_name.capitalize(), 'Team'])
                drop_dups = drop_dups.dropna(subset=[col_name.capitalize()])
                drop_dups = drop_dups.dropna(axis=1, how='all')
                
                seven_day = pd.DataFrame()
                for i, row in drop_dups.iterrows():
                    
                    # Query the DataFrame for matching rows
                    batting = row['Batting']
                    team = row['Team']
                    matching_rows = combined_results[(combined_results['Batting'] == batting.strip()) & (combined_results['Team'] == team)]
                    match_desc = matching_rows.describe()
                    # Append the matching rows to the combined results DataFrame
                    seven_day = pd.concat([seven_day,matching_rows])
                date = delta.days
                break
        
