from simulation.market_sim import MarketSimulation
from simulation.portfolio import Portfolio


class ANSI:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"



def ordinal(n):
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return str(n) + suffix



def prettyPrintPortfolio(simulation: MarketSimulation, username: str):
    pfDict = simulation.getPortfolioValueAtCurrentDate(username)

    print(f"{ANSI.BOLD}{ANSI.CYAN}Portfolio for {username}:{ANSI.RESET}")
    print(f"Cash: {ANSI.BOLD}{ANSI.GREEN if pfDict['cash'] > 0 else ''}${pfDict['cash']:,.2f}{ANSI.RESET}")
    if len(pfDict["positions"]) > 0:
        print("Positions:")
    else:
        print("No positions currently held.")

    for ticker, posData in pfDict["positions"].items():
        quantity = posData["quantity"]
        currentPrice = posData["currentPrice"]
        positionValue = posData["value"]
        averagePrice = posData["averagePrice"]
        absReturn = posData["absReturn"]
        pctReturn = posData["pctReturn"]
        returnColor = (ANSI.GREEN if absReturn > 0 else ANSI.RED if absReturn < 0 else ANSI.YELLOW) + ANSI.BOLD

        prettyQuantity = f"{quantity:.3f}".rstrip('0').rstrip('.')
        print(f"  {ticker}: {prettyQuantity} shares @ ${currentPrice:,.2f} = {ANSI.BOLD}{ANSI.GREEN}${positionValue:,.2f}{ANSI.RESET}")
        print(f"    ⤷ average price: ${averagePrice:,.2f}")
        print(f"    ⤷ return: {returnColor}{absReturn:+,.2f}{ANSI.RESET} ({returnColor}{pctReturn:+.2f}%{ANSI.RESET})")     
    
    portfolioAbsReturn = pfDict["absReturn"]
    portfolioPctReturn = pfDict["pctReturn"]

    returnColor = (ANSI.GREEN if portfolioAbsReturn > 0 else ANSI.RED if portfolioAbsReturn < 0 else ANSI.YELLOW) + ANSI.BOLD
    print(f"\nTotal Portfolio Value: {ANSI.BOLD}${pfDict['totalValue']:,.2f}{ANSI.RESET}")
    print(  f" ⤷ Return: {returnColor}{portfolioAbsReturn:+,.2f}{ANSI.RESET} ({returnColor}{portfolioPctReturn:+.2f}%{ANSI.RESET})")
    print("=" * 80)
    print("")

