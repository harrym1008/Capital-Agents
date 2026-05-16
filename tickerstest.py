from collectors.tickerclient import TickerDataClient
from collectors.constants import *

if __name__ == "__main__":
    client = TickerDataClient(START_DATE, END_DATE)
    client.massTickerDownloadWithData()