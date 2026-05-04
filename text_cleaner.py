#human written code
import unicodedata

def keep(c):
    if c.isascii():
        return True
    return unicodedata.category(c).startswith(('S', 'P'))

with open('pasta.txt', 'r', encoding='utf-8') as f:
    cleaned_data = [
        ''.join(c for c in line if keep(c))
        for line in f
    ]

with open('cleaned_pasta.txt', 'w', encoding='utf-8') as f:
    f.writelines(cleaned_data)