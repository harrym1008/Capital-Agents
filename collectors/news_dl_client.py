import os
from datetime import datetime, timezone, timedelta
import threading
import concurrent.futures

# Alpaca's direct Python SDK has issues with pagination, so I will use raw requests
import requests
import re
import html
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from collectors.rate_limiter import GlobalRateLimiters
from collectors.constants import (
    NEWS_BATCHES_DIR,
    NEWS_PARQUET_PATH,
    NEWS_BATCH_SIZE,
    NEWS_ROW_GROUP_SIZE,
)
from collectors.news_cleaner import (
    cleanHtmlContent, removeDuplicateHeadline, removeBenzingaFooter,
    FILTER_PIPELINE
)


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



def processChunk(chunk):
    # Make a copy of the chunk to avoid modifying the original DataFrame in place
    chunk = chunk.copy()

    chunk["wordCount"] = chunk["content"].apply(lambda x: len(str(x).split()))
    chunk["content"] = chunk["content"].apply(cleanHtmlContent)
    chunk["content"] = chunk.apply(
        lambda row: removeDuplicateHeadline(row["headline"], row["content"]), axis=1
    )
    chunk["content"] = chunk["content"].apply(removeBenzingaFooter)

    chunk["removeReason"] = None
    for filterFunc, _reason in FILTER_PIPELINE:
        mask = chunk["removeReason"].isna()
        chunk.loc[mask, "removeReason"] = chunk[mask].apply(filterFunc, axis=1)

    return chunk


def cleanAndFilterArticlesDf(df):
    # Parallelise the CPU-bound cleaning + filtering across every core
    numWorkers = os.cpu_count() or 1
    if len(df) < numWorkers * 10:
        # Too small to bother spawning processes
        results = [processChunk(df)]
    else:
        chunks = np.array_split(df, numWorkers)
        with concurrent.futures.ProcessPoolExecutor(max_workers=numWorkers) as executor:
            results = list(executor.map(processChunk, chunks))

    processed = pd.concat(results, ignore_index=True)

    # Drop duplicate headlines (keep the earliest occurrence).
    processed = processed.drop_duplicates(subset=["headline"], keep="first")

    badDf = processed[processed["removeReason"].notna()].copy()
    goodDf = processed[processed["removeReason"].isna()].copy()

    badDf.to_parquet("data/newsfiltered.parquet", engine="pyarrow", index=False)
    return goodDf.drop(columns=["removeReason", "wordCount"])



class NewsClient:
    def __init__(self, startDate, endDate, rateLimiterDatabase: GlobalRateLimiters):
        load_dotenv()
        self.apiKey = os.getenv("ALPACA_API_KEY_2")
        self.apiSecret = os.getenv("ALPACA_API_SECRET_2")
        self.alpacaLimiter = rateLimiterDatabase.alpacaNewsDlLimiter

        self.startDate = datetime.strptime(startDate, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        self.endDate = datetime.strptime(endDate, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, microsecond=999999, tzinfo=timezone.utc
        )       

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

        # Track progress by the latest article timestamp, the total num of articles is not known
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

        # Worker function for each thread to fetch news articles in its assigned time window
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
                                "date": article.get("updated_at", ""),
                                "headline": article.get("headline", ""),
                                "content": article.get("content", ""),
                                "author": article.get("author", ""),
                                "tickers": article.get("symbols", [])
                            })

                        if articlesList:
                            with sharedLock:
                                totalFetched += len(articlesList)

                        latestArticleTime = articlesList[-1].get("updated_at") if articlesList else None
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
                            df["date"] = pd.to_datetime(df["date"])

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
                df["date"] = pd.to_datetime(df["date"])

                batchFileName = f"{NEWS_BATCHES_DIR}/news_{threadId}_{batchCount:03d}.parquet"
                df.to_parquet(batchFileName, engine="pyarrow", index=False)

                batchCount += 1
                with sharedLock:
                    totalBatchCount += 1

            refreshProgress(threadId, int(windowEnd.timestamp()))

        # Initiate a worker for every single window
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
            consolidatedDf = cleanAndFilterArticlesDf(consolidatedDf)

            # Sort the consolidated DataFrame by "date" to maintain stability
            consolidatedDf = consolidatedDf.sort_values("date", kind="mergesort")
            self.writeNewsParquet(consolidatedDf, NEWS_PARQUET_PATH)
        else:
            print("Could not download any news articles! News parquet file will be empty.")
            emptyDf = pd.DataFrame(columns=["id", "date", "headline", "content", "author", "tickers"])
            self.writeNewsParquet(emptyDf, NEWS_PARQUET_PATH)

        # Clear the batch files after consolidation
        for filename in os.listdir(NEWS_BATCHES_DIR):
            filePath = os.path.join(NEWS_BATCHES_DIR, filename)
            if os.path.isfile(filePath):
                os.remove(filePath)
        # os.rmdir(NEWS_BATCHES_DIR)

        return totalFetched, totalBatchCount


    # Write the final consolidated DataFrame to a parquet file
    def writeNewsParquet(self, df, path):
        if len(df) == 0:
            table = pa.table({
                "id": pa.array([], type=pa.string()),
                "date": pa.array([], type=pa.timestamp("us")),
                "headline": pa.array([], type=pa.string()),
                "content": pa.array([], type=pa.string()),
                "author": pa.array([], type=pa.string()),
                "tickers": pa.array([], type=pa.list_(pa.string())),
            })
        else:
            tickers = []
            for s in df["tickers"].tolist():
                tickers.append(list(s) if s is not None else [])

            table = pa.table({
                "id": pa.array(
                    [str(x) if x is not None else None for x in df["id"].tolist()],
                    type=pa.string()
                ),
                "date": pa.array(df["date"].tolist(), type=pa.timestamp("us")),
                "headline": pa.array(df["headline"].tolist(), type=pa.string()),
                "content": pa.array(df["content"].tolist(), type=pa.string()),
                "author": pa.array(df["author"].tolist(), type=pa.string()),
                "tickers": pa.array(tickers, type=pa.list_(pa.string())),
            })

        pq.write_table(table, path, row_group_size=NEWS_ROW_GROUP_SIZE, compression="snappy")



    

