
import exchange_calendars as xcals


class MarketCalendar:
    def __init__(self, startDate, endDate):
        self.nyseCalendar = xcals.get_calendar("XNYS")
        
        sessions = self.nyseCalendar.sessions_in_range(startDate, endDate)
        self.openDays = set(sessions.strftime("%Y-%m-%d"))


    def isOpenDay(self, date):
        return date in self.openDays