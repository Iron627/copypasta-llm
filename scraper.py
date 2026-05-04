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
seen = set()

def scrape_page(url):
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"Request error {url}: {e}")
        return

    soup = BeautifulSoup(r.text, 'html.parser')
    pasta = soup.find_all('code')

    with open('pasta.txt', 'a', encoding='utf-8') as f:
        for p in pasta:
            text = p.text.strip()
            if text not in seen:
                seen.add(text)
                f.write(text + "\n\n")
for i in range(1,n):
    scrape_page(f"https://copypastatext.com/page/{i}")
    time.sleep(RATE_LIMIT)