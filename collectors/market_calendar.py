import exchange_calendars as xcals

class MarketCalendar:
    def __init__(self, startDate, endDate):
        self.nyseCalendar = xcals.get_calendar("XNYS")

        firstValid = self.nyseCalendar.first_session.strftime("%Y-%m-%d")
        lastValid = self.nyseCalendar.last_session.strftime("%Y-%m-%d")
        clippedStart = max(startDate, firstValid)
        clippedEnd = min(endDate, lastValid)

        sessions = self.nyseCalendar.sessions_in_range(clippedStart, clippedEnd)
        self.openDays = set(sessions.strftime("%Y-%m-%d"))

    def isOpenDay(self, date):
        return date in self.openDays
