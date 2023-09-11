import requests
from bs4 import BeautifulSoup
import pandas as pd

base_url = "https://www.baseball-reference.com"
url = base_url+"/leagues/majors/2023-schedule.shtml"

response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

boxscore_links = []
for link in soup.find_all("a"):
    if link.text == "Boxscore":
        boxscore_links.append(base_url + link.get("href"))

dataframes = []
for link in boxscore_links:
    response = requests.get(base_url+link)
    soup = BeautifulSoup(response.content, "html.parser")
    table = soup.find("table", {"id": "boxscore"})
    df = pd.read_html(str(table))[0]
    dataframes.append(df)

print(dataframes)