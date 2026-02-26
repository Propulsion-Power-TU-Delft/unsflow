total_chars = 100
total_chars_mid = total_chars//2

def print_banner(string=''):
    n = total_chars - 2
    print("+", f"{string:-^{n}}", "+", sep='')
