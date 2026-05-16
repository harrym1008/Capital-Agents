import pandas as pd
from dateutil import tz

NEW_YORK = tz.gettz("America/New_York")

START_DATE_STR = "2016-01-01"
START_DATE = pd.Timestamp(START_DATE_STR, tz=NEW_YORK)

FIRST_TRAD_DAY_AFTER_START_STR = "2016-01-04"
FIRST_TRAD_DAY_AFTER_START = pd.Timestamp(FIRST_TRAD_DAY_AFTER_START_STR, tz=NEW_YORK)

END_DATE_STR = "2026-04-30"
END_DATE = pd.Timestamp(END_DATE_STR, tz=NEW_YORK)

IPO_BEFORE_START_DATE = pd.Timestamp("2015-12-31", tz=NEW_YORK)

