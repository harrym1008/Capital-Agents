import os
from datetime import datetime, timezone, timedelta
import threading

# Alpaca's direct Python SDK has issues with pagination, so I will use raw requests
import requests
import re
import html
import ast
from dotenv import load_dotenv

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from collectors.rate_limiter import GlobalRateLimiters
from collectors.constants import (
    NEWS_BATCHES_DIR,
    NEWS_PARQUET_PATH,
    NEWS_INDEX_PARQUET_PATH,
    NEWS_BATCH_SIZE,
    NEWS_ROW_GROUP_SIZE,
)


def cleanupArticleContent(rawContent):
    if not rawContent:
        return ""
    text = re.sub(r"<.*?>", "", rawContent)  # Remove HTML tags
    text = html.unescape(text)  # Convert HTML entities to characters
    text = " ".join(text.split())  # Clean whitespace
    return text


def splitDateRange(startDate, endDate, threadCount):
    totalSeconds = int((endDate - startDate).total_seconds())
    if threadCount <= 1 or totalSeconds <= 0:
        return [(startDate, endDate)]

    windowSize = totalSeconds / threadCount
    ranges = []

    for i in range(threadCount):
        windowStart = startDate + timedelta(seconds=int(i * windowSize))
        if i == threadCount - 1:
            windowEnd = endDate
        else:
            windowEnd = startDate + timedelta(seconds=int((i + 1) * windowSize))
        ranges.append((windowStart, windowEnd))

    return ranges



class NewsClient:
    def __init__(self, startDate, endDate, rateLimiterDatabase: GlobalRateLimiters):
        load_dotenv()
        self.apiKey = os.getenv("ALPACA_API_KEY")
        self.apiSecret = os.getenv("ALPACA_API_SECRET")
        self.alpacaLimiter = rateLimiterDatabase.alpacaLimiter

        self.startDate = datetime.strptime(startDate, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        self.endDate = datetime.strptime(endDate, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        


    def threadedMassDownload(self, threads=4, pbar=None):
        os.makedirs(NEWS_BATCHES_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(NEWS_PARQUET_PATH), exist_ok=True)

        # Delete old files
        if os.path.exists(NEWS_BATCHES_DIR):
            for filename in os.listdir(NEWS_BATCHES_DIR):
                os.remove(os.path.join(NEWS_BATCHES_DIR, filename))

        if os.path.exists(NEWS_PARQUET_PATH):
            os.remove(NEWS_PARQUET_PATH)

        url = "https://data.alpaca.markets/v1beta1/news"
        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": self.apiKey,
            "APCA-API-SECRET-KEY": self.apiSecret
        }

        windows = splitDateRange(self.startDate, self.endDate, threads)
        threadStartUnix = [int(window[0].timestamp()) for window in windows]
        threadProgress = threadStartUnix.copy()

        totalFetched = 0
        totalBatchCount = 0
        sharedLock = threading.Lock()

        startUnix = int(self.startDate.timestamp())
        endUnix = int(self.endDate.timestamp())

        if pbar is None:
            pbar = tqdm(
                total=endUnix - startUnix,
                desc="Fetching news articles",
                smoothing=0.1,
                bar_format="{desc} | {percentage:3.2f}% |{bar}| [{elapsed} elapsed, {remaining} remaining] ",
                colour="green",
                dynamic_ncols=True
            )
        else:
            pbar.reset(total=endUnix - startUnix)
            pbar.set_description("News: Fetching")

        def refreshProgress(threadId, latestUnix):
            nonlocal totalFetched, totalBatchCount
            with sharedLock:
                threadProgress[threadId] = max(threadProgress[threadId], latestUnix)
                progressValue = 0
                for i in range(len(threadProgress)):
                    progressValue += max(0, threadProgress[i] - threadStartUnix[i])

                maxUnix = max(threadProgress)
                latestDate = datetime.fromtimestamp(maxUnix, tz=timezone.utc).strftime("%#d %B %Y")
                pbar.set_description(
                    f"({totalFetched} | {totalBatchCount * NEWS_BATCH_SIZE}) {latestDate} | Fetching news articles"
                )
                pbar.n = progressValue
                pbar.refresh()

        def worker(threadId, windowStart, windowEnd):
            nonlocal totalFetched, totalBatchCount

            batchBuffer = []
            batchCount = 0
            nextPageToken = None

            while True:
                params = {
                    "start": windowStart.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "end": windowEnd.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "limit": 50,
                    "sort": "asc",
                    "include_content": True
                }
                if nextPageToken:
                    params["page_token"] = nextPageToken

                iters = 0
                pageSuccess = False

                while iters < 10:
                    self.alpacaLimiter.wait()

                    try:
                        response = requests.get(url, headers=headers, params=params)

                        if response.status_code == 429:
                            iters += 1
                            self.alpacaLimiter.got429(iters)
                            continue

                        response.raise_for_status()
                        newsResponse = response.json()

                        articlesList = newsResponse.get("news", [])

                        for article in articlesList:
                            articleId = article.get("id")

                            batchBuffer.append({
                                "id": articleId,
                                "updated_at": article.get("updated_at", ""),
                                "headline": article.get("headline", ""),
                                "content": cleanupArticleContent(article.get("content", "")),
                                "author": article.get("author", ""),
                                # Store as a real list (not a stringified one) so the
                                # parquet column becomes LIST<STRING> and DuckDB can use
                                # list_contains() instead of slow string matching.
                                "symbols": article.get("symbols", [])
                            })

                        if articlesList:
                            with sharedLock:
                                totalFetched += len(articlesList)

                        latestArticleTime = articlesList[-1]["updated_at"] if articlesList else None
                        if latestArticleTime:
                            latestArticleDt = datetime.strptime(latestArticleTime, "%Y-%m-%dT%H:%M:%SZ").replace(
                                tzinfo=timezone.utc
                            )
                            latestArticleUnix = int(latestArticleDt.timestamp())
                            refreshProgress(threadId, latestArticleUnix)

                        nextPageToken = newsResponse.get("next_page_token", None)
                        pageSuccess = True

                        if len(batchBuffer) >= NEWS_BATCH_SIZE:
                            df = pd.DataFrame(batchBuffer)
                            df["updated_at"] = pd.to_datetime(df["updated_at"])

                            batchFileName = f"{NEWS_BATCHES_DIR}/news_{threadId}_{batchCount:03d}.parquet"
                            df.to_parquet(batchFileName, engine="pyarrow", index=False)

                            batchCount += 1
                            with sharedLock:
                                totalBatchCount += 1
                            batchBuffer = []

                        break

                    except Exception as e:
                        iters += 1
                        self.alpacaLimiter.non429Error(e)

                if not pageSuccess:
                    break

                if not nextPageToken:
                    break

            if batchBuffer:
                df = pd.DataFrame(batchBuffer)
                df["updated_at"] = pd.to_datetime(df["updated_at"])

                batchFileName = f"{NEWS_BATCHES_DIR}/news_{threadId}_{batchCount:03d}.parquet"
                df.to_parquet(batchFileName, engine="pyarrow", index=False)

                batchCount += 1
                with sharedLock:
                    totalBatchCount += 1

            refreshProgress(threadId, int(windowEnd.timestamp()))

        threads = []
        for threadId, (windowStart, windowEnd) in enumerate(windows):
            thread = threading.Thread(
                target=worker,
                args=(threadId, windowStart, windowEnd),
                daemon=True
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        if pbar is not None and not hasattr(pbar, '_external'):
            pbar.close()

        batchFiles = sorted([
            f for f in os.listdir(NEWS_BATCHES_DIR)
            if f.endswith(".parquet")
        ])

        tempFiles = []
        for index, filename in enumerate(batchFiles):
            sourcePath = os.path.join(NEWS_BATCHES_DIR, filename)
            tempName = f"temp_{index:06d}.parquet"
            tempPath = os.path.join(NEWS_BATCHES_DIR, tempName)
            os.rename(sourcePath, tempPath)
            tempFiles.append(tempPath)

        for index, tempPath in enumerate(tempFiles):
            finalPath = os.path.join(NEWS_BATCHES_DIR, f"news_{index:03d}.parquet")
            os.rename(tempPath, finalPath)

        # Consolidate all batch files into a single parquet file
        batchFiles = sorted([
            os.path.join(NEWS_BATCHES_DIR, f)
            for f in os.listdir(NEWS_BATCHES_DIR)
            if f.endswith(".parquet")
        ])

        if batchFiles:
            dfs = [pd.read_parquet(f, engine="pyarrow") for f in batchFiles]
            consolidatedDf = pd.concat(dfs, ignore_index=True)
            # Sort by date so each row group covers a contiguous time range. This lets
            # DuckDB skip entire row groups via min/max statistics on `updated_at`
            # (row-group pruning) for date-range queries.
            consolidatedDf = consolidatedDf.sort_values("updated_at", kind="mergesort")
            _write_news_parquet(consolidatedDf, NEWS_PARQUET_PATH)
        else:
            emptyDf = pd.DataFrame(columns=["id", "updated_at", "headline", "content", "author", "symbols"])
            _write_news_parquet(emptyDf, NEWS_PARQUET_PATH)

        return totalFetched, totalBatchCount


def _write_news_parquet(df, path):
    """Write the news dataframe to parquet with an explicit Arrow schema.

    Guarantees `symbols` is stored as LIST<STRING> (so DuckDB can use
    list_contains) and uses a tuned row-group size for date-range pruning.
    """
    if len(df) == 0:
        table = pa.table({
            "id": pa.array([], type=pa.string()),
            "updated_at": pa.array([], type=pa.timestamp("us")),
            "headline": pa.array([], type=pa.string()),
            "content": pa.array([], type=pa.string()),
            "author": pa.array([], type=pa.string()),
            "symbols": pa.array([], type=pa.list_(pa.string())),
        })
    else:
        # Normalise symbols to a list of strings even if a row somehow holds a
        # stringified list (e.g. reading legacy batch files).
        symbols = []
        for s in df["symbols"].tolist():
            if isinstance(s, str):
                try:
                    s = ast.literal_eval(s)
                except Exception:
                    s = []
            symbols.append(list(s) if s is not None else [])

        table = pa.table({
            "id": pa.array(df["id"].tolist(), type=pa.string()),
            "updated_at": pa.array(df["updated_at"].tolist(), type=pa.timestamp("us")),
            "headline": pa.array(df["headline"].tolist(), type=pa.string()),
            "content": pa.array(df["content"].tolist(), type=pa.string()),
            "author": pa.array(df["author"].tolist(), type=pa.string()),
            "symbols": pa.array(symbols, type=pa.list_(pa.string())),
        })

    pq.write_table(table, path, row_group_size=NEWS_ROW_GROUP_SIZE, compression="snappy")


    def buildInvertedIndex(self, pbar=None):
        if not os.path.exists(NEWS_PARQUET_PATH):
            return

        df = pd.read_parquet(NEWS_PARQUET_PATH, engine="pyarrow", columns=["id", "symbols"])
        indices = {}

        if pbar is None:
            pbar = tqdm(
                total=len(df),
                desc="Building inverted index",
                smoothing=0.1,
                bar_format="{desc} | {percentage:3.2f}% |{bar}| [{elapsed} elapsed, {remaining} remaining] ",
                colour="green",
                dynamic_ncols=True
            )
        else:
            pbar.reset(total=len(df))
            pbar.set_description("News: Building index")

        for row in df.itertuples(index=False):
            articleId = row.id
            symbolsVal = row.symbols
            # New format: symbols is already a list (LIST<STRING> column).
            # Legacy format: a stringified list -> fall back to literal_eval.
            if isinstance(symbolsVal, str):
                try:
                    symbolsList = ast.literal_eval(symbolsVal)
                except Exception:
                    symbolsList = []
            elif symbolsVal is None:
                symbolsList = []
            else:
                symbolsList = list(symbolsVal)

            for symbol in symbolsList:
                if symbol not in indices:
                    indices[symbol] = []
                indices[symbol].append(articleId)

            pbar.update(1)

        if pbar is not None and not hasattr(pbar, '_external'):
            pbar.close()
        out = pd.DataFrame(
            {"symbol": list(indices.keys()), "ids": list(indices.values())}
        ).sort_values("symbol")

        out.to_parquet(NEWS_INDEX_PARQUET_PATH, engine="pyarrow", index=False)

