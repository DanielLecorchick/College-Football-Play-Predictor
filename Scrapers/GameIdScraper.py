"""
College Football Play Prediction - Game ID Scraper
Author: Trenton Ottman
Purpose: Get all game IDs from ESPNs website
"""

import requests
from bs4 import BeautifulSoup
import re
import time
from typing import Set, List

"""
Scraper for ESPN game IDs for college football games.
    
Args:
    weeks: List of week numbers to scrape (e.g., [1, 2, 3, ..., 16])
    groups: List of conference groups (1=ACC, 4=Big Ten, 5=Big 12, 8=SEC)
    year: Season year (default: 2024)
    seasontype: 2 for regular season, 3 for postseason
    
Returns:
    Set of unique game IDs
"""

def scrape_espn_game_ids(weeks: List[int], groups: List[int], year: int = 2024, seasontype: int = 2) -> Set[str]:

    base_url = "https://www.espn.com/college-football/schedule/_/week/{week}/year/{year}/seasontype/{seasontype}/group/{group}"
    
    game_ids = set()
    total_requests = len(weeks) * len(groups)
    current_request = 0
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for week in weeks:
        for group in groups:
            current_request += 1
            url = base_url.format(week=week, year=year, seasontype=seasontype, group=group)
            
            print(f"[{current_request}/{total_requests}] Scraping Week {week}, Group {group}...")
            
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                game_links = soup.find_all('a', href=re.compile(r'/college-football/game/_/gameId/\d+')) #gets all links that contain game IDs
                
                for link in game_links:
                    href = link.get('href')
                    match = re.search(r'/gameId/(\d+)', href) # extract game ID using regex
                    if match:
                        game_id = match.group(1)
                        game_ids.add(game_id)
                
                time.sleep(0.5)
                
            except requests.RequestException as e:
                print(f"  Error fetching {url}: {e}")
                continue
    
    return game_ids


def main():
    #parameters for Power 4 conferences
    weeks = list(range(1, 17))  #weeks 1-16
    groups = [1, 4, 5, 8]  #ACC, Big Ten, Big 12, SEC (power 4 confrences)
    
    print("Starting ESPN College Football Game ID Scraper")
    print("=" * 60)
    print(f"Scraping weeks: {min(weeks)} to {max(weeks)}")
    print(f"Conference groups: {groups}")
    print("=" * 60)
    print()
    
    #gets all regular season games
    game_ids = scrape_espn_game_ids(weeks=weeks, groups=groups, year=2024, seasontype=2)
    
    print()
    print("=" * 60)
    print(f"Total unique game IDs found: {len(game_ids)}")
    print("=" * 60)
    print()
    
    # Sort and display results
    sorted_ids = sorted(game_ids)
    for game_id in sorted_ids:
        print(game_id)
    
    #Save to file
    output_file = "espn_game_ids_2024.txt"
    with open(output_file, 'w') as f:
        for game_id in sorted_ids:
            f.write(f"{game_id}\n")
    
    print()
    print(f"Game IDs saved to {output_file}")


if __name__ == "__main__":
    main()