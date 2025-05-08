from cgi import print_environ_usage
from types import NoneType
from urllib import response
import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
import time
import io

### baseball reference 
# boxscore_links.index(link)
# 1754
# len(boxscore_links)
# 2358
base_url = "https://www.baseball-reference.com"
url = base_url+"/leagues/majors/2024-schedule.shtml"#"/leagues/MLB-schedule.shtml"#

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:110.0) Gecko/20100101 Firefox/110.0",
    "Referer": "https://www.baseball-reference.com/",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Connection": "keep-alive"
}

#response = requests.get(url, headers=headers)
#soup = BeautifulSoup(response.content, "html5lib")
# session = requests.Session()
# session.headers.update(headers)
# response = session.get(url)
import subprocess

# At the top of your script, add a helper function
def remove_unnamed_columns(df):
    """Remove any columns with 'Unnamed' in their name"""
    return df.loc[:, ~df.columns.str.contains('Unnamed', case=False)]

def http_request(curl_url: str):
    # Define the curl command
    curl_command = [
        "curl",
        "-X", "GET",
        curl_url,
        "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:110.0) Gecko/20100101 Firefox/110.0",
        "-H", "Referer: https://www.baseball-reference.com/",
        "-H", "Accept-Language: en-US,en;q=0.9",
        "-H", "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "-H", "Connection: keep-alive"
    ]

    # Execute the curl command and capture the output
    result = subprocess.run(curl_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Store the response in a variable
    response_content = result.stdout

    # Print the response content
    return response_content

response_content = http_request(url)
soup = BeautifulSoup(response_content, "html5lib")

boxscore_links = []
for link in soup.find_all("a"):
    if link.text == "Boxscore":
        boxscore_links.append(base_url + link.get("href"))

dataframes = []
complete_batting = pd.DataFrame()
complete_pitching = pd.DataFrame()
complete_linescore = pd.DataFrame()
batting = pd.DataFrame()
pitching = pd.DataFrame()
prev_date = '-1'
for link in boxscore_links:
    # if boxscore_links.index(link) < 1754:
    #     continue
    path = link.split('/')
    date = path[-1]
    file_name = date[3:-6]
    if prev_date not in link: 
        if batting.empty is False and len(prev_date) > 4:
            path = link.split('/')
            date = path[-1]
            file_name = date[3:-6]
            batting['Date'] = prev_date
            pitching['Date'] = prev_date
            batting = batting.dropna(subset=['AB'])
            batting = batting.drop(batting[batting['Batting'] == 'Team Totals'].index)
            pitching = pitching.drop(pitching[pitching['Pitching'] == 'Team Totals'].index)
            batting.to_csv(prev_date+'_batting.csv')
            pitching.to_csv(prev_date+'_pitching.csv')

            complete_batting = pd.concat([complete_batting, batting], ignore_index=True)
            complete_pitching = pd.concat([complete_pitching, pitching], ignore_index=True)
            
            with pd.ExcelWriter(prev_date+'.xlsx', mode='w') as xwriter:
                batting.to_excel(xwriter, sheet_name='batting')
                pitching.to_excel(xwriter, sheet_name='pitching')
            batting = pd.DataFrame()
            pitching = pd.DataFrame()

        prev_date = file_name
    
    print('Downloading {}'.format(link))
    start_time = time.time()
    resp = http_request(link)#requests.get(link, headers=headers)
    box = BeautifulSoup(resp, "html5lib")

    
    # Look for the linescore table in the main HTML (not in comments)
    linescore_div = box.find('div', class_='linescore_wrap')
    if linescore_div:
        linescore_table = linescore_div.find('table', class_='linescore')
        if linescore_table:
            teams = []
            rows = linescore_table.find('tbody').find_all('tr')
            
            for row in rows:
                # The team name is in the second td element, inside an anchor tag
                team_cell = row.find_all('td')[1]
                team_link = team_cell.find('a')
                if team_link:
                    team_name = team_link.text.strip()
                    teams.append(team_name)
            
            # Now teams[0] is the away team and teams[1] is the home team
            away_team_name = teams[0] if len(teams) > 0 else None
            home_team_name = teams[1] if len(teams) > 1 else None


            linescore_df = pd.read_html(io.StringIO(str(linescore_table)))[0]
            # Process the linescore data as needed
            linescore_df['Date'] = prev_date
            linescore_df['AwayTeam'] = away_team_name
            linescore_df['HomeTeam'] = home_team_name
            linescore_df.to_csv(prev_date+'_linescore.csv')
            complete_linescore = pd.concat([complete_linescore, linescore_df], ignore_index=True)
            print(f"Saved linescore for {prev_date}")

    
    comments = box.find_all(string=lambda text: isinstance(text, Comment))

#<team>batting - add HR, 2B, 3B, SB, GDP
#<team>pitching
    for comment in comments:
        comment_soup = BeautifulSoup(comment, "html5lib")
        
        # Extract lineups from the main page (not in comments)
        lineup_div = comment_soup.find('div', id='div_lineups')
        if lineup_div:
            lineups = {}
            lineup_tables = lineup_div.find_all('table')
            
            for table in lineup_tables:
                caption = table.find('caption')
                if caption:
                    team_name = caption.text.strip()
                    lineup = []
                    
                    rows = table.find('tbody').find_all('tr')
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 3:
                            # Get batting order number (empty for pitcher)
                            order_num = cells[0].text.strip()
                            
                            # Get player name
                            player_cell = cells[1]
                            player_link = player_cell.find('a')
                            player_name = player_link.text.strip() if player_link else "Unknown"
                            player_id = player_link['href'].split('/')[-1].split('.')[0] if player_link else ""
                            
                            # Get position
                            position = cells[2].text.strip()
                            
                            lineup.append({
                                'order': order_num,
                                'player_name': player_name,
                                'player_id': player_id,
                                'position': position,
                                'team': team_name
                            })
                    
                    lineups[team_name] = lineup
            
            # Convert lineups to DataFrame and save
            if lineups:
                lineup_df = pd.DataFrame()
                for team, players in lineups.items():
                    team_df = pd.DataFrame(players)
                    lineup_df = pd.concat([lineup_df, team_df], ignore_index=True)
                
                lineup_df['Date'] = prev_date
                lineup_df['Home'] = lineup_df['team'].apply(lambda x: home_team_name in x or x in home_team_name)
                
                # Save lineup data
                lineup_df.to_csv(prev_date + '_lineup.csv')
                
                # If you want to add to a complete dataframe
                if 'complete_lineup' not in locals():
                    complete_lineup = pd.DataFrame()
                complete_lineup = pd.concat([complete_lineup, lineup_df], ignore_index=True)
                print(f"Saved lineup for {prev_date}")


        for tag in comment_soup.find_all("table"):
            class_name = tag.get("class")
            if class_name != None:
                if 'stats_table' in class_name:
                    team = tag.find('caption').text.replace('Table', '').strip()
                    df = pd.read_html(io.StringIO(str(tag)))[0]
                    # Set the Home flag based on team name
                    is_home_team = (team == home_team_name)
                    if 'Batting' in df.columns:
                        df = df.assign(Team=team,HR=0, Double=0, Triple=0, SB=0, GDP=0, Home=is_home_team)
                        for d, drow in df.iterrows():
                            details = drow['Details']
                            if 'float' in str(type(details)):
                                continue
                            items = details.split(',')
                            for item in items:
                                num = 1
                                if len(item) > 3:
                                    num = eval(item[0])
                                    
                                if item.find('HR') > -1:
                                    df.loc[d,'HR'] += num
                                elif item.find('Double') > -1:
                                    df.loc[d,'Double'] += num
                                elif item.find('Triple') > -1:
                                    df.loc[d,'Triple'] += num
                                elif item.find('SB') > -1:                           
                                    df.loc[d,'SB'] += num
                                elif item.find('GDP') > -1:
                                    df.loc[d,'GDP'] += num
                                    
                        batting = pd.concat([batting, df], ignore_index=True)
                    elif 'Pitching' in df.columns:
                        df = df.assign(Team=team, W=0, L=0, HLD=0, SV=0, Home=is_home_team,Throws='R')
                        ##loop over df and get W,L,H,S
                        for p, prow in df.iterrows():
                            prow['Team'] = team
                            pitcher = prow['Pitching']
                            pitch_details = pitcher.split(',')

                            if len(pitch_details) > 1:
                                stat = pitch_details[-1].strip()
                                if 'BS' in stat:
                                    continue
                                elif 'W' in stat:
                                    df.loc[p,'W'] += 1
                                elif 'H' in stat:
                                    df.loc[p,'HLD'] += 1
                                elif 'L' in stat:
                                    df.loc[p,'L'] += 1
                                elif 'S' in stat:
                                    df.loc[p,'SV'] += 1
                                                    
                        pitching = pd.concat([pitching, df], ignore_index=True)
                        #have to make another call to get throws R/L
    end_time = time.time()
    elapsed_time = end_time - start_time
    while elapsed_time < 5:
        print("sleeping...")
        time.sleep(1)
        end_time = time.time()
        elapsed_time = end_time - start_time
                        
                    
# Clean up all dataframes before saving
complete_batting = remove_unnamed_columns(complete_batting)
complete_pitching = remove_unnamed_columns(complete_pitching)
complete_linescore = remove_unnamed_columns(complete_linescore)
if 'complete_lineup' in locals():
    complete_lineup = remove_unnamed_columns(complete_lineup)
    complete_lineup.to_csv('2024_lineups.csv', index=False)

# Save CSV files
complete_batting.to_csv('2024_batting.csv', index=False)
complete_pitching.to_csv('2024_pitching.csv', index=False)
complete_linescore.to_csv('2024_linescore.csv', index=False)

# Update Excel writer to include linescore and lineups
with pd.ExcelWriter('2024_stats.xlsx', mode='w') as xwriter:
    complete_batting.to_excel(xwriter, sheet_name='batting', index=False)
    complete_pitching.to_excel(xwriter, sheet_name='pitching', index=False)
    complete_linescore.to_excel(xwriter, sheet_name='linescore', index=False)
    if 'complete_lineup' in locals():
        complete_lineup.to_excel(xwriter, sheet_name='lineups', index=False)