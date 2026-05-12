import os

from simulation.marketsim import MarketSimulation
from simulation.orders import MarketOrder, StopOrder, LimitOrder, StopLimitOrder, OrderSide
from simulation.prettysim import prettyPrintPortfolio, ANSI, ordinal


if __name__ == "__main__":
    sim = MarketSimulation("2018-01-01", "2026-05-09")
    sim.initialiseUsers(["NVDA", "SNDK", "RYCEY", "MNTS", "SPY", "MU", "INTC"], testingRun=True)    

    for i in range(100001):
        os.system("cls" if os.name == "nt" else "clear")
        print(f"{ANSI.BOLD}Day {i}: {sim.currentDate.strftime(f'%A %#d %B %Y')}{ANSI.RESET}")
        print("=" * 80)
        print()
        
        sim.prettyPrintAllPortfolios()

        # input("\n\nPress Enter to continue...  ") 

        # import time
        # time.sleep(0.01)

        if not sim.runNextDay():
            break
        

    sim.concludeSimulation()

    os.system("cls" if os.name == "nt" else "clear")
    print(f"{ANSI.BOLD}Final Day: {sim.currentDate.strftime('%A %#d %B %Y')}{ANSI.RESET}\n")
    sim.prettyPrintAllPortfolios()