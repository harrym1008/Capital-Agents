import os
from dotenv import load_dotenv
from collectors.dailynews import NewsClient
from collectors.ratelimiter import GlobalRateLimiters

def main():
    load_dotenv()

    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")

    if not api_key or not api_secret:
        print("Error: ALPACA_API_KEY or ALPACA_API_SECRET not found in environment.")
        return

    if input("ARE YOU SURE? This will download ALL news articles from 1 Jan 2016 to 30 Apr 2026 and WILL take MULTIPLE HOURS! (yes/no) ").lower() != "yes":
        print("Aborting.")
        return

    client = NewsClient(api_key, api_secret, "2016-01-01", "2026-04-30", GlobalRateLimiters())
    client.massDownloadNews()



if __name__ == "__main__":
    main()
