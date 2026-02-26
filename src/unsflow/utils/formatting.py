# used throughout the tool
total_chars = 100
total_chars_mid = total_chars//2

def print_banner_begin(string):
    """
    Prints the initial banner.
    :param string: string to include in the banner, in central position
    """
    n = total_chars - 2
    print("+", f"{string:-^{n}}", "+", sep='')


def print_banner_end(string=''):
    """
    Prints the final banner.
    """
    n = total_chars - 2
    print("+", f"{string:-^{n}}", "+", sep='')
    print()