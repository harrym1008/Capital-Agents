import os
import json

from data.tickers import TickerClient, Exchange
from data.companyprofile import DescriptionClient, CompanyProfile
from data.dailyprices import DailyPriceClient



if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    tickerClient = TickerClient(os.getenv("FINNHUB_API_KEY"))
    descriptionClient = DescriptionClient(os.getenv("FINNHUB_API_KEY"))
    dailyPriceClient = DailyPriceClient()

    allSecurities = tickerClient.getAllTickers()
    nyseSecurities = tickerClient.getFormattedSecuritiesForExchange(allSecurities, Exchange.NYSE)
    nasdaqSecurities = tickerClient.getFormattedSecuritiesForExchange(allSecurities, Exchange.NASDAQ)
    # etfSecurities = tickerClient.getFormattedSecuritiesForExchange(allSecurities, Exchange.ETF)

    print(f"Total NYSE Securities: {len(nyseSecurities)}")
    print(f"Total NASDAQ Securities: {len(nasdaqSecurities)}")
    # print(f"Total ETF Securities: {len(etfSecurities)}")

    exportData = {}

    for (exchange, securitiesList) in [(Exchange.NYSE, nyseSecurities), (Exchange.NASDAQ, nasdaqSecurities)]:  # , (Exchange.ETF, etfTickers)]:
        print(f"\n{exchange} Securities:")
        profile = []
        for security in securitiesList[:16]:
            ticker = security["ticker"]
            description = descriptionClient.getDescription(ticker)
            prices, latestClosePrice = dailyPriceClient.getDailyPrices(ticker)
            prettyName = description.prettyName if len(description.prettyName) < 24 else description.prettyName[:21] + "..."

            data = {
                "ticker": ticker,
                "description": description,
                "prices": prices
            }
            profile.append(data)

            priceText = f"{f'${latestClosePrice:.2f}':>8}"
            print(f"{ticker:<7}Last close: {priceText}\t{prettyName:<24} - {description.industry} --> {description.sector}")
        print(" done.")


    # outputFilename = "db/ticker.json"
    # with open(outputFilename, "w") as f:
    #     json.dump(exportData, f, indent=4)

    
    

