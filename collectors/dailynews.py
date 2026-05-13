import os
from datetime import datetime, timezone, timedelta
import threading

# Alpaca's direct Python SDK has issues with pagination, so I will use raw requests
import requests
import re
import html
import pandas as pd
from tqdm import tqdm

from collectors.ratelimiter import GlobalRateLimiters

NEWS_THREADS = 4


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
    def __init__(self, apiKey, apiSecret, startDate, endDate, rateLimiterDatabase: GlobalRateLimiters):
        self.apiKey = apiKey
        self.apiSecret = apiSecret
        self.alpacaLimiter = rateLimiterDatabase.alpacaLimiter

        self.startDate = datetime.strptime(startDate, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        self.endDate = datetime.strptime(endDate, "%Y-%m-%d").replace(tzinfo=timezone.utc)


    def massDownloadNews(self):
        batchesDir = "data/newsbatches"
        parquetPath = "data/news.parquet"
        os.makedirs(batchesDir, exist_ok=True)
        os.makedirs(os.path.dirname(parquetPath), exist_ok=True)
        
        # Delete old files
        if os.path.exists(batchesDir):
            for filename in os.listdir(batchesDir):
                os.remove(os.path.join(batchesDir, filename))
            
        if os.path.exists(parquetPath):
            os.remove(parquetPath)
        
        batchSize = 2000
        
        batchBuffer = []
        batchCount = 0
        totalFetched = 0
        
        url = "https://data.alpaca.markets/v1beta1/news"
        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": self.apiKey,
            "APCA-API-SECRET-KEY": self.apiSecret
        }
        
        nextPageToken = None
        
        # Keep track of progress with tqdm using the date of the last article fetched vs the total date range
        startUnix = int(self.startDate.timestamp())
        endUnix = int(self.endDate.timestamp())
        pbar = tqdm(
            total=endUnix - startUnix, 
            desc="Fetching news articles",
            smoothing=0.8,
            bar_format="{desc}| {percentage:3.2f}% |{bar}| [{elapsed} elapsed, {remaining} remaining] "         
        )
        
        while True:
            params = {
                "start": self.startDate.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end": self.endDate.strftime("%Y-%m-%dT%H:%M:%SZ"),
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
                            "symbols": str(article.get("symbols", []))
                        })
                        totalFetched += 1

                    latestArticleTime = articlesList[-1]["updated_at"] if articlesList else None
                    if latestArticleTime:
                        latestArticleDt = datetime.strptime(latestArticleTime, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                        latestArticleUnix = int(latestArticleDt.timestamp())
                        latestDate = latestArticleDt.strftime("%#d %B %Y")

                        pbar.set_description(f"({totalFetched} | {batchCount * batchSize}) {latestDate} | Fetching news articles")
                        pbar.n = latestArticleUnix - startUnix
                        pbar.refresh()
                    
                    nextPageToken = newsResponse.get("next_page_token", None)
                    pageSuccess = True
                    
                    # Write batch if full
                    if len(batchBuffer) >= batchSize:
                        df = pd.DataFrame(batchBuffer)
                        df['updated_at'] = pd.to_datetime(df['updated_at'])
                        
                        batchFileName = f"{batchesDir}/news_{batchCount:03d}.parquet"
                        df.to_parquet(batchFileName, engine="pyarrow", index=False)
                        
                        batchCount += 1
                        batchBuffer = []
                    
                    break
                
                except Exception as e:
                    iters += 1
                    self.alpacaLimiter.non429Error(e)
            
            if not pageSuccess:
                print(f"\nFAILED to fetch news page after 10 attempts!")
                break
            
            if not nextPageToken:
                print("\nNo more pages. All articles fetched.")
                break
        
        pbar.close()

        # Write remaining buffer
        if batchBuffer:
            df = pd.DataFrame(batchBuffer)
            df['updated_at'] = pd.to_datetime(df['updated_at'])
            
            batchFileName = f"{batchesDir}/news_{batchCount:03d}.parquet"
            df.to_parquet(batchFileName, engine="pyarrow", index=False)
            
            batchCount += 1
            batchBuffer = []
        
        print(f"\nDone! Total fetched: {totalFetched}, Total batches saved: {batchCount}")
        
        # Consolidate all batch files into a single parquet file
        print(f"\nConsolidating {batchCount} batch files into {parquetPath}...")
        batchFiles = sorted([
            os.path.join(batchesDir, f) 
            for f in os.listdir(batchesDir) 
            if f.endswith(".parquet")
        ])
        
        if batchFiles:
            dfs = [pd.read_parquet(f, engine="pyarrow") for f in batchFiles]
            consolidatedDf = pd.concat(dfs, ignore_index=True)
            consolidatedDf.to_parquet(parquetPath, engine="pyarrow", index=False)
            print(f"Consolidated {len(consolidatedDf)} articles into {parquetPath}")
        else:
            # Create empty parquet file if no batches
            emptyDf = pd.DataFrame(columns=["id", "updated_at", "headline", "content", "author", "symbols"])
            emptyDf.to_parquet(parquetPath, engine="pyarrow", index=False)
            print("No articles to consolidate. Created empty parquet file.")
        
        return totalFetched, batchCount


    def threadedMassDownload(self):
        threadCount = NEWS_THREADS

        batchesDir = "data/newsbatches"
        parquetPath = "data/news.parquet"
        os.makedirs(batchesDir, exist_ok=True)
        os.makedirs(os.path.dirname(parquetPath), exist_ok=True)

        # Delete old files
        if os.path.exists(batchesDir):
            for filename in os.listdir(batchesDir):
                os.remove(os.path.join(batchesDir, filename))

        if os.path.exists(parquetPath):
            os.remove(parquetPath)

        batchSize = 2000

        url = "https://data.alpaca.markets/v1beta1/news"
        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": self.apiKey,
            "APCA-API-SECRET-KEY": self.apiSecret
        }

        windows = splitDateRange(self.startDate, self.endDate, threadCount)
        threadStartUnix = [int(window[0].timestamp()) for window in windows]
        threadProgress = threadStartUnix.copy()

        totalFetched = 0
        totalBatchCount = 0
        sharedLock = threading.Lock()

        startUnix = int(self.startDate.timestamp())
        endUnix = int(self.endDate.timestamp())
        pbar = tqdm(
            total=endUnix - startUnix,
            desc="Fetching news articles",
            smoothing=0.8,
            bar_format="{desc}| {percentage:3.2f}% |{bar}| [{elapsed} elapsed, {remaining} remaining] "
        )

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
                    f"({totalFetched} | {totalBatchCount * batchSize}) {latestDate} | Fetching news articles"
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
                                "symbols": str(article.get("symbols", []))
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

                        if len(batchBuffer) >= batchSize:
                            df = pd.DataFrame(batchBuffer)
                            df["updated_at"] = pd.to_datetime(df["updated_at"])

                            batchFileName = f"{batchesDir}/news_{threadId}_{batchCount:03d}.parquet"
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
                    print(f"\nFAILED to fetch news page after 10 attempts for thread {threadId}!")
                    break

                if not nextPageToken:
                    print(f"\nThread {threadId} completed. No more pages.")
                    break

            if batchBuffer:
                df = pd.DataFrame(batchBuffer)
                df["updated_at"] = pd.to_datetime(df["updated_at"])

                batchFileName = f"{batchesDir}/news_{threadId}_{batchCount:03d}.parquet"
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

        pbar.close()

        print(f"\nDone! Total fetched: {totalFetched}, Total batches saved: {totalBatchCount}")

        print(f"\nRenaming batch files...")
        batchFiles = sorted([
            f for f in os.listdir(batchesDir)
            if f.endswith(".parquet")
        ])

        tempFiles = []
        for index, filename in enumerate(batchFiles):
            sourcePath = os.path.join(batchesDir, filename)
            tempName = f"temp_{index:06d}.parquet"
            tempPath = os.path.join(batchesDir, tempName)
            os.rename(sourcePath, tempPath)
            tempFiles.append(tempPath)

        for index, tempPath in enumerate(tempFiles):
            finalPath = os.path.join(batchesDir, f"news_{index:03d}.parquet")
            os.rename(tempPath, finalPath)

        # Consolidate all batch files into a single parquet file
        print(f"\nConsolidating {totalBatchCount} batch files into {parquetPath}...")
        batchFiles = sorted([
            os.path.join(batchesDir, f)
            for f in os.listdir(batchesDir)
            if f.endswith(".parquet")
        ])

        if batchFiles:
            dfs = [pd.read_parquet(f, engine="pyarrow") for f in batchFiles]
            consolidatedDf = pd.concat(dfs, ignore_index=True)
            consolidatedDf.to_parquet(parquetPath, engine="pyarrow", index=False)
            print(f"Consolidated {len(consolidatedDf)} articles into {parquetPath}")
        else:
            emptyDf = pd.DataFrame(columns=["id", "updated_at", "headline", "content", "author", "symbols"])
            emptyDf.to_parquet(parquetPath, engine="pyarrow", index=False)
            print("No articles to consolidate. Created empty parquet file.")

        return totalFetched, totalBatchCount
