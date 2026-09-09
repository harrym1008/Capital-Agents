import os
import threading
import requests
import duckdb
import pandas as pd
from dotenv import load_dotenv

from collectors.constants import NEWS_PARQUET_PATH, UTC, IPO_BEFORE_START_DATE
from collectors.rate_limiter import GlobalRateLimiters
from collectors.news_dl_client import cleanAndFilterArticlesDf
from dataquery.lru_cache import LRUCache

load_dotenv()


class NewsDataProvider:
    def __init__(self, cache: LRUCache, rateLimiters: GlobalRateLimiters):
        self.cache = cache
        self.rateLimiters = rateLimiters
        self.con = duckdb.connect(database=":memory:")
        self.lock = threading.RLock()
        self.maxLocalDate = self.getMaxLocalDate()

    def normaliseTimestamp(self, before: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(before)
        if ts.tzinfo is None:
            ts = ts.tz_localize(UTC)
        else:
            ts = ts.tz_convert(UTC)
        return ts.tz_localize(None)

    def getMaxLocalDate(self) -> pd.Timestamp:
        if os.path.exists(NEWS_PARQUET_PATH):
            try:
                res = self.con.execute(f"SELECT MAX(date) FROM read_parquet('{NEWS_PARQUET_PATH}')").fetchone()
                if res and res[0] is not None:
                    self.maxLocalDate = pd.to_datetime(res[0])
                    return self.maxLocalDate
            except Exception:
                pass
        self.maxLocalDate = pd.Timestamp(IPO_BEFORE_START_DATE).tz_localize(None)
        return self.maxLocalDate


    def downloadNonLocalNews(self, startDate: pd.Timestamp, endDate: pd.Timestamp, tickers: list[str] = None):
        apiKeyId = os.getenv("ALPACA_API_KEY_2") or os.getenv("ALPACA_API_KEY")
        apiKeySecret = os.getenv("ALPACA_API_SECRET_2") or os.getenv("ALPACA_API_SECRET")
        if not apiKeyId or not apiKeySecret:
            return

        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": apiKeyId,
            "APCA-API-SECRET-KEY": apiKeySecret
        }

        params = {
            "start": startDate.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": endDate.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": 50,
            "sort": "asc",
            "include_content": "true",
            "exclude_contentless": "true",
        }
        if tickers:
            params["symbols"] = ",".join([t.upper() for t in tickers if t])

        url = "https://data.alpaca.markets/v1beta1/news"
        nextPageToken = None
        fetched = []

        while True:
            if nextPageToken:
                params["page_token"] = nextPageToken

            self.rateLimiters.alpacaNewsDlLimiter.wait()

            try:
                resp = requests.get(url, headers=headers, params=params, timeout=10)
                if resp.status_code != 200:
                    break
                newsData = resp.json()
                articles = newsData.get("news", [])
                for art in articles:
                    artId = str(art.get("id", ""))
                    rawDate = art.get("updated_at") or art.get("created_at") or art.get("date")
                    parsedDate = pd.to_datetime(rawDate).tz_convert("UTC").tz_localize(None) if rawDate else endDate
                    headline = art.get("headline", "")
                    content = art.get("content", "")
                    author = art.get("author", "").strip()
                    symbols = art.get("symbols", [])

                    fetched.append({
                        "id": artId,
                        "date": parsedDate,
                        "headline": headline,
                        "content": content,
                        "author": author,
                        "tickers": symbols
                    })

                nextPageToken = newsData.get("next_page_token")
                if not nextPageToken or len(articles) == 0:
                    break
            except Exception:
                break

        if fetched:
            dfInc = pd.DataFrame(fetched)
            dfClean = cleanAndFilterArticlesDf(dfInc)
            if not dfClean.empty:
                self.con.register("temp_inc", dfClean)
                self.con.execute("CREATE TABLE IF NOT EXISTS inc_news_table AS SELECT * FROM temp_inc WHERE 1=0")
                self.con.execute("INSERT INTO inc_news_table SELECT * FROM temp_inc WHERE id NOT IN (SELECT id FROM inc_news_table)")
                self.con.unregister("temp_inc")


    def ensureCoverage(self, targetTimestamp: pd.Timestamp, tickers: list[str] = None):
        maxLocal = self.getMaxLocalDate()
        if targetTimestamp > maxLocal:
            with self.lock:
                self.downloadNonLocalNews(maxLocal, targetTimestamp, tickers=tickers)
                self.maxLocalDate = targetTimestamp


    def getQueryRelationSql(self) -> str:
        hasInc = False
        try:
            res = self.con.execute("SELECT COUNT(*) FROM inc_news_table").fetchone()
            if res and res[0] > 0:
                hasInc = True
        except Exception:
            hasInc = False

        if hasInc:
            if os.path.exists(NEWS_PARQUET_PATH):
                return f"(SELECT * FROM read_parquet('{NEWS_PARQUET_PATH}') UNION ALL SELECT * FROM inc_news_table)"
            else:
                return "inc_news_table"
        else:
            if os.path.exists(NEWS_PARQUET_PATH):
                return f"read_parquet('{NEWS_PARQUET_PATH}')"
            else:
                return None

    def truncateDfContent(self, df: pd.DataFrame, summaryMaxChars: int) -> pd.DataFrame:
        if df is None or df.empty or "content" not in df.columns:
            return df
        if summaryMaxChars is not None and summaryMaxChars > 0:
            df = df.copy()
            def truncateText(text):
                if not text:
                    return text
                strText = str(text)
                if len(strText) > summaryMaxChars and not strText.endswith("... [content truncated]"):
                    return strText[:summaryMaxChars] + "... [content truncated]"
                return strText
            df["content"] = df["content"].apply(truncateText)
        return df

    def getRecentNewsForTicker(self, ticker: str, before: pd.Timestamp, 
                               limit: int = 12, 
                               mustHaveContent: bool = False, 
                               maxReferencedTickers: int = 5,
                               summaryMaxChars: int = 2500) -> pd.DataFrame:
        if limit < 1:
            return pd.DataFrame()

        ticker = ticker.upper()
        beforeNorm = self.normaliseTimestamp(before)

        with self.lock:
            key = f"news|single_{ticker}_{beforeNorm.strftime('%Y-%m-%dH%H')}_{limit}_{mustHaveContent}_{maxReferencedTickers}_{summaryMaxChars}"
            cached = self.cache.get(key)
            if isinstance(cached, pd.DataFrame):
                return cached

            self.ensureCoverage(beforeNorm, tickers=[ticker])

            relationSql = self.getQueryRelationSql()
            if relationSql is None:
                return pd.DataFrame()

            if summaryMaxChars is not None and summaryMaxChars > 0:
                if mustHaveContent:
                    contentCond = "COALESCE(LENGTH(content), 0) > 0 AND contains(substr(content, 1, ?), ?)"
                    contentParams = [int(summaryMaxChars), ticker]
                else:
                    contentCond = "(COALESCE(LENGTH(content), 0) = 0 OR contains(substr(content, 1, ?), ?))"
                    contentParams = [int(summaryMaxChars), ticker]
            else:
                if mustHaveContent:
                    contentCond = "COALESCE(LENGTH(content), 0) > 0 AND contains(content, ?)"
                    contentParams = [ticker]
                else:
                    contentCond = "(COALESCE(LENGTH(content), 0) = 0 OR contains(content, ?))"
                    contentParams = [ticker]

            sql = f"""
                SELECT *
                FROM {relationSql}
                WHERE date < ?
                    AND list_contains(tickers, ?)
                    AND {contentCond}
                    {"AND array_length(tickers) <= ?" if maxReferencedTickers is not None else ""}
                ORDER BY date DESC, id
                LIMIT ?
            """
            queryParams = [beforeNorm.to_pydatetime(), ticker] + contentParams
            if maxReferencedTickers is not None:
                queryParams.append(int(maxReferencedTickers))
            queryParams.append(int(limit))
            try:
                df = self.con.execute(sql, queryParams).df()
            except Exception:
                df = pd.DataFrame()

            df = self.truncateDfContent(df, summaryMaxChars)
            self.cache.put(key, df)
            return df

    def getRecentNewsForTickers(self, tickers: list[str], before: pd.Timestamp, 
                               limit: int = 12, 
                               mustHaveContent: bool = False, 
                               maxReferencedTickers: int = 5,
                               summaryMaxChars: int = 2500,
                               onProgressCallback=None) -> pd.DataFrame:
        if limit < 1:
            return pd.DataFrame()

        tickerSet = sorted({t.upper() for t in tickers if t})
        if not tickerSet:
            return pd.DataFrame()

        beforeNorm = self.normaliseTimestamp(before)

        with self.lock:
            key = f"news|multi_{','.join(tickerSet)}_{beforeNorm.strftime('%Y-%m-%dH%H')}_{limit}_{mustHaveContent}_{maxReferencedTickers}_{summaryMaxChars}"
            cached = self.cache.get(key)
            if isinstance(cached, pd.DataFrame):
                return cached

            self.ensureCoverage(beforeNorm, tickers=tickerSet)

            relationSql = self.getQueryRelationSql()
            if relationSql is None:
                return pd.DataFrame()

            conditions = [
                "date < ?",
                "list_has_any(tickers, ?)",
            ]

            queryParams = [
                beforeNorm.to_pydatetime(),
                tickerSet,
            ]

            if maxReferencedTickers is not None:
                conditions.append("array_length(tickers) <= ?")
                queryParams.append(int(maxReferencedTickers))

            if mustHaveContent:
                conditions.append("COALESCE(LENGTH(content), 0) > 0")

            if summaryMaxChars is not None and summaryMaxChars > 0:
                conditions.append(
                    "len(list_filter(?, t -> contains(substr(content, 1, ?), t))) > 0"
                )
                queryParams.extend([tickerSet, int(summaryMaxChars)])

            sql = f"""
                SELECT *
                FROM {relationSql}
                WHERE {" AND ".join(conditions)}
                ORDER BY date DESC, id
                LIMIT ?
            """

            queryParams.append(int(limit))

            try:
                df = self.con.execute(sql, queryParams).df()
            except Exception:
                df = pd.DataFrame()

            df = self.truncateDfContent(df, summaryMaxChars)
            self.cache.put(key, df)

            return df

    def getNewsForTickerBetweenTimes(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp, 
                                     mustHaveContent: bool = False, 
                                     maxReferencedTickers: int = 5,
                                     summaryMaxChars: int = 2500) -> pd.DataFrame:
        ticker = ticker.upper()
        startNorm = self.normaliseTimestamp(start)
        endNorm = self.normaliseTimestamp(end)

        with self.lock:
            key = f"news|range_{ticker}_{startNorm.strftime('%Y-%m-%dH%H')}_{endNorm.strftime('%Y-%m-%dH%H')}_{mustHaveContent}_{maxReferencedTickers}_{summaryMaxChars}"
            cached = self.cache.get(key)
            if isinstance(cached, pd.DataFrame):
                return cached

            self.ensureCoverage(endNorm, tickers=[ticker])

            relationSql = self.getQueryRelationSql()
            if relationSql is None:
                return pd.DataFrame()

            if summaryMaxChars is not None and summaryMaxChars > 0:
                if mustHaveContent:
                    contentCond = "COALESCE(LENGTH(content), 0) > 0 AND contains(substr(content, 1, ?), ?)"
                    contentParams = [int(summaryMaxChars), ticker]
                else:
                    contentCond = "(COALESCE(LENGTH(content), 0) = 0 OR contains(substr(content, 1, ?), ?))"
                    contentParams = [int(summaryMaxChars), ticker]
            else:
                if mustHaveContent:
                    contentCond = "COALESCE(LENGTH(content), 0) > 0 AND contains(content, ?)"
                    contentParams = [ticker]
                else:
                    contentCond = "(COALESCE(LENGTH(content), 0) = 0 OR contains(content, ?))"
                    contentParams = [ticker]

            sql = f"""
                SELECT *
                FROM {relationSql}
                WHERE date >= ? AND date <= ?
                    AND list_contains(tickers, ?)
                    AND {contentCond}
                    {"AND array_length(tickers) <= ?" if maxReferencedTickers is not None else ""}
                ORDER BY date DESC, id
            """
            queryParams = [startNorm.to_pydatetime(), endNorm.to_pydatetime(), ticker] + contentParams
            if maxReferencedTickers is not None:
                queryParams.append(int(maxReferencedTickers))
            try:
                df = self.con.execute(sql, queryParams).df()
            except Exception:
                df = pd.DataFrame()

            df = self.truncateDfContent(df, summaryMaxChars)
            self.cache.put(key, df)
            return df

    def getNewsForTickersBetweenTimes(self, tickers: list[str], start: pd.Timestamp, end: pd.Timestamp, 
                                      mustHaveContent: bool = False, 
                                      maxReferencedTickers: int = 5,
                                      summaryMaxChars: int = 2500) -> pd.DataFrame:
        tickerSet = sorted({t.upper() for t in tickers if t})
        if not tickerSet:
            return pd.DataFrame()

        startNorm = self.normaliseTimestamp(start)
        endNorm = self.normaliseTimestamp(end)

        with self.lock:
            key = f"news|multirange_{','.join(tickerSet)}_{startNorm.strftime('%Y-%m-%dH%H')}_{endNorm.strftime('%Y-%m-%dH%H')}_{mustHaveContent}_{maxReferencedTickers}_{summaryMaxChars}"
            cached = self.cache.get(key)
            if isinstance(cached, pd.DataFrame):
                return cached

            self.ensureCoverage(endNorm, tickers=tickerSet)

            relationSql = self.getQueryRelationSql()
            if relationSql is None:
                return pd.DataFrame()

            if summaryMaxChars is not None and summaryMaxChars > 0:
                if mustHaveContent:
                    contentCond = "COALESCE(LENGTH(content), 0) > 0 AND len(list_filter(?, t -> contains(substr(content, 1, ?), t))) > 0"
                    contentParams = [tickerSet, int(summaryMaxChars)]
                else:
                    contentCond = "(COALESCE(LENGTH(content), 0) = 0 OR len(list_filter(?, t -> contains(substr(content, 1, ?), t))) > 0)"
                    contentParams = [tickerSet, int(summaryMaxChars)]
            else:
                if mustHaveContent:
                    contentCond = "COALESCE(LENGTH(content), 0) > 0 AND len(list_filter(?, t -> contains(content, t))) > 0"
                    contentParams = [tickerSet]
                else:
                    contentCond = "(COALESCE(LENGTH(content), 0) = 0 OR len(list_filter(?, t -> contains(content, t))) > 0)"
                    contentParams = [tickerSet]

            sql = f"""
                SELECT *
                FROM {relationSql}
                WHERE date >= ? AND date <= ?
                    AND list_has_any(tickers, ?)
                    AND {contentCond}
                    {"AND array_length(tickers) <= ?" if maxReferencedTickers is not None else ""}
                ORDER BY date DESC, id
            """
            queryParams = [startNorm.to_pydatetime(), endNorm.to_pydatetime(), tickerSet] + contentParams
            if maxReferencedTickers is not None:
                queryParams.append(int(maxReferencedTickers))
            try:
                df = self.con.execute(sql, queryParams).df()
            except Exception:
                df = pd.DataFrame()

            df = self.truncateDfContent(df, summaryMaxChars)
            self.cache.put(key, df)
            return df

