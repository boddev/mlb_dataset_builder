from cgi import print_environ_usage
from types import NoneType
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
url = base_url+"/leagues/majors/2023-schedule.shtml"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, "html5lib")

boxscore_links = []
for link in soup.find_all("a"):
    if link.text == "Boxscore":
        boxscore_links.append(base_url + link.get("href"))

dataframes = []
batting = pd.DataFrame()
pitching = pd.DataFrame()
prev_date = '-1'
for link in boxscore_links:
    if boxscore_links.index(link) < 1754:
        continue
    path = link.split('/')
    date = path[-1]
    file_name = date[3:-6]
    if prev_date not in link: 
        if batting.empty is False and len(prev_date) > 4:
            path = link.split('/')
            date = path[-1]
            file_name = date[3:-6]
            batting = batting.dropna(subset=['AB'])
            batting = batting.drop(batting[batting['Batting'] == 'Team Totals'].index)
            pitching = pitching.drop(pitching[pitching['Pitching'] == 'Team Totals'].index)
            batting.to_csv(prev_date+'_batting.csv')
            pitching.to_csv(prev_date+'_pitching.csv')
            with pd.ExcelWriter(prev_date+'.xlsx', mode='w') as xwriter:
                batting.to_excel(xwriter, sheet_name='batting')
                pitching.to_excel(xwriter, sheet_name='pitching')
            batting = pd.DataFrame()
            pitching = pd.DataFrame()

        prev_date = file_name
    
    print('Downloading {}'.format(link))
    start_time = time.time()
    resp = requests.get(link, headers=headers)
    box = BeautifulSoup(resp.content, "html5lib")
    comments = box.find_all(string=lambda text: isinstance(text, Comment))

#<team>batting - add HR, 2B, 3B, SB, GDP
#<team>pitching
    for comment in comments:
        comment_soup = BeautifulSoup(comment, "html5lib")
        for tag in comment_soup.find_all("table"):
            class_name = tag.get("class")
            if class_name != None:
                if 'stats_table' in class_name:
                    team = tag.find('caption').text.replace('Table', '').strip()
                    df = pd.read_html(io.StringIO(str(tag)))[0]
                    if 'Batting' in df.columns:
                        df = df.assign(Team=team,HR=0, Double=0, Triple=0, SB=0, GDP=0)
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
                        df = df.assign(Team=team, W=0, L=0, HLD=0, SV=0, Throws='R')
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
                        
                    
                    #dataframes.append(df)

print(dataframes)