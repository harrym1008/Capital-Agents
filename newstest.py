
from collectors.dailynews import NewsClient
from collectors.ratelimiter import GlobalRateLimiters
from collectors.constants import *

def main():
    if input("ARE YOU SURE? This will download ALL news articles from 1 Jan 2016 to 30 Apr 2026 and WILL take MULTIPLE HOURS! (yes/no) ").lower() != "yes":
        print("Aborting.")
        return

    client = NewsClient(START_DATE_STR, END_DATE_STR, GlobalRateLimiters())
    client.threadedMassDownload()



if __name__ == "__main__":
    main()
