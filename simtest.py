import os
import time

from simulation.marketsim import MarketSimulation
from simulation.orders import MarketOrder, StopOrder, LimitOrder, StopLimitOrder, OrderSide
from simulation.prettysim import prettyPrintPortfolio, ANSI, ordinal

from collectors.constants import START_DATE_STR, END_DATE_STR

if __name__ == "__main__":
    sim = MarketSimulation("2016-06-01", END_DATE_STR)
    sim.initialiseUsers(["NVDA", "INTC", "AAPL", "MSFT", "BINI"], testingRun=True)    


    for i in range(100001):
        # Only print on the first day of the month
        if sim.currentDate.day == 1 or i == 0:
            os.system("cls" if os.name == "nt" else "clear")
            print("\033[2J\033[H", end="")  # clear screen, move cursor to top-left
            print(f"{ANSI.BOLD}Day {i}: {sim.currentDate.strftime(f'%A %#d %B %Y')}{ANSI.RESET}")
            print("=" * 80)
            print()
            

            sim.prettyPrintAllPortfolios()
            time.sleep(0.1)

        if not sim.runNextDay():
            break
        

    sim.concludeSimulation()

    print("\033[2J\033[H", end="")
    print(f"{ANSI.BOLD}Final Day: {sim.currentDate.strftime('%A %#d %B %Y')}{ANSI.RESET}")
    print("=" * 80)
    sim.prettyPrintAllPortfolios()