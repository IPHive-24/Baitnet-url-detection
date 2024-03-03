import pandas as pd

def main():
    blacklist_path = r"C:\Users\manoj\OneDrive\Desktop\FINAL\verified_online (1).csv"
    whitelist_path = r"C:\Users\manoj\OneDrive\Desktop\FINAL\whitelist.txt"
    urls = {}
    
    blacklist_df = pd.read_csv(blacklist_path)
    whitelist = []
    
    with open(whitelist_path, 'r') as f:
        lines = f.read().splitlines()
        whitelist.extend(lines)

    # Assign 0 for non-malicious and 1 as malicious for supervised learning.
    for url in blacklist_df['url']:
        urls[url] = 1
    
    for url in whitelist:
        urls[url] = 0

    return urls

if __name__ == "__main__":
    main()
