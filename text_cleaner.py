#human written code
import string

allowed = set(string.ascii_lowercase + string.ascii_uppercase + " @$&*#.,!?'\n0123456789")

with open('pasta.txt', 'r', encoding='utf-8') as f:
    cleaned_data = [
        ''.join(c for c in line if c in allowed)
        for line in f
    ]

with open('cleaned_pasta.txt', 'w', encoding='utf-8') as f:
    f.writelines(cleaned_data)