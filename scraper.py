# human written code.
import requests
from bs4 import BeautifulSoup
import time
RATE_LIMIT = 5
n = 100

base_url = "https://copypastatext.com/page/1"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Referer': 'https://google.com'
}
def scrape_page(url):
    r = requests.get(url,headers=headers)
    txt = r.text
    soup = BeautifulSoup(txt,'html.parser')
    pasta = soup.find_all('code')
    with open('pasta.txt','a') as f:
        for p in pasta:
            f.write(p.text)
            f.write('\n\n')
    print(f"Scraped {url}")
for i in range(1,n):
    scrape_page(f"https://copypastatext.com/page/{i}")
    time.sleep(RATE_LIMIT)