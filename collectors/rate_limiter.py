import time
import math
import threading
from typing import Optional, Dict, Any


BURST_RATIO = 0.75
SLOW_MULTIPLIER = 1 / (1 - BURST_RATIO)   # = 4.0
MIN_INTERVAL = 0.1


class RateLimiter:
    def __init__(self, name: str, maxCalls: int, period: float = 60.0):
        self.name = name
        self.maxCalls = max(1, int(maxCalls))
        self.period = float(period)

        # Mathematical parameters
        self.callInterval = self.period / self.maxCalls
        self.burstQuota = int(self.maxCalls * BURST_RATIO)

        # Thread synchronization and state tracking
        self.lock = threading.Lock()
        self.lastCallTime = 0.0
        self.nextAllowedTime = 0.0

        # Unix boundary tracking
        self.currentWindowStart = self.calculateWindowStart(time.time())
        self.callsInWindow = 0


    def calculateWindowStart(self, timestamp: float) -> float:
        # Returns the aligned timestamp of the start of the current window for a given timestamp
        return math.floor(timestamp / self.period) * self.period


    def syncWindow(self, now: float) -> None:
        # Synchronises the current window state based on the provided timestamp
        window_start = self.calculateWindowStart(now)
        if window_start > self.currentWindowStart:
            # Unix boundary crossed: reset window counter
            self.currentWindowStart = window_start
            self.callsInWindow = 0


    def getWaitTime(self) -> float:
        # Returns the estimated wait time in seconds for a subsequent caller invoking wait() at this moment
        
        with self.lock:
            now = time.time()
            self.syncWindow(now)

            backoffWait = max(0.0, self.nextAllowedTime - now)

            if self.callsInWindow >= self.maxCalls:
                # Window quota is exhausted... must wait until next exact Unix boundary
                nextWindow = self.currentWindowStart + self.period
                boundaryWait = max(0.0, nextWindow - now)
                return max(backoffWait, boundaryWait)

            # We can make the call within the current window
            if self.callsInWindow < self.burstQuota:
                # Fast lane with minimal interval
                pacingWait = max(0.0, (self.lastCallTime + MIN_INTERVAL) - now)
            else:
                # Throttled lane
                slowInterval = max(MIN_INTERVAL, self.callInterval * SLOW_MULTIPLIER)
                pacingWait = max(0.0, (self.lastCallTime + slowInterval) - now)

            return max(backoffWait, pacingWait)


    def wait(self) -> float:
        # Blocks the caller until the next allowed time that an API call can be made
        startTime = time.time()

        while True:
            sleepDuration = 0.0
            with self.lock:
                now = time.time()
                self.syncWindow(now)

                # Check if currently held under 429 exponential backoff
                if now < self.nextAllowedTime:
                    sleepDuration = self.nextAllowedTime - now

                # Check if current Unix epoch window quota is maxed out
                elif self.callsInWindow >= self.maxCalls:
                    # Sleep until the exact microsecond the next Unix window starts
                    nextBoundary = self.currentWindowStart + self.period
                    sleepDuration = max(0.001, nextBoundary - now)

                else:
                    # Quota is available, determine interval depending on burst tier
                    if self.callsInWindow < self.burstQuota:
                        requiredInterval = MIN_INTERVAL
                    else:
                        requiredInterval = self.callInterval * SLOW_MULTIPLIER

                    # Earliest time this call may proceed
                    targetTime = max(now, self.lastCallTime + requiredInterval)
                    nextWindow = self.currentWindowStart + self.period


                    if targetTime >= nextWindow and self.callsInWindow > 0:
                        # Scheduled call would exceed current window, so wait for next window instead
                        sleepDuration = max(0.001, nextWindow - now)
                    else:
                        # 
                        # Successfully reserve the execution slot
                        self.callsInWindow += 1
                        self.lastCallTime = targetTime
                        sleepDuration = max(0.0, targetTime - now)

                        # Sleep outside the lock if needed, then proceed
                        if sleepDuration > 0:
                            time.sleep(sleepDuration)
                        return time.time() - startTime

            # Sleep outside lock if window was full or under 429 backoff, then re-check
            if sleepDuration > 0:
                time.sleep(sleepDuration)


    def calculate429WaitTime(self, iters: int = 0) -> float:
        # Calculate exponentially increasing back off time after first or consecutive 429 responses
        waitTime = min(round(7.5 * (1.5 ** iters)), self.period * 2)  # Cap at double the period
        return waitTime
    

    def got429(self, iters: int = 0) -> float:
        # Handle 429 error
        waitTime = self.calculate429WaitTime(iters)
        print(f"[{self.name}] Received 429 \"Too Many Requests\". Waiting for {waitTime:.2f} seconds...")
        
        with self.lock:
            backoffUntil = time.time() + waitTime
            if backoffUntil > self.nextAllowedTime:
                self.nextAllowedTime = backoffUntil
                
        time.sleep(waitTime)
        return waitTime


    def non429Error(self, errOrTime: Exception | float) -> None:
        if isinstance(errOrTime, Exception):
            print(f"[{self.name}] Received non-429 error - {errOrTime.__class__.__name__}: {errOrTime}. Waiting for 15 seconds before retrying...")
            time.sleep(15)
        else:
            # Handle case where a float wait time is passed instead of an exception
            print(f"[{self.name}] Received non-429 error - wait time: {errOrTime:.2f} seconds. Waiting before retrying...")
            time.sleep(errOrTime)
        

    def getStats(self) -> Dict[str, Any]:
        """
        Returns telemetry snapshot of the rate limiter's internal state.
        """
        with self.lock:
            now = time.time()
            self.syncWindow(now)
            next_window = self.currentWindowStart + self.period
            return {
                "name": self.name,
                "currentWindowStart": self.currentWindowStart,
                "nextWindowStart": next_window,
                "secondsUntilRefresh": round(max(0.0, next_window - now), 3),
                "callsInWindow": self.callsInWindow,
                "maxCalls": self.maxCalls,
                "burstQuota": self.burstQuota,
                "remainingCalls": max(0, self.maxCalls - self.callsInWindow),
                "isBurstPhase": self.callsInWindow < self.burstQuota,
            }



class GlobalRateLimiters:
    def __init__(self):
        self.yFinanceLimiter = RateLimiter("yfinance", 100, 60)     # Not an official limit, but should be safe
        self.alpacaLimiter = RateLimiter("alpaca", 200, 60)         # Alpaca free allows 200 requests per minute
        self.finnhubLimiter = RateLimiter("finnhub", 60, 60)        # Finnhub free allows 60 requests per minute
        self.massiveLimiter = RateLimiter("massive", 5, 60)         # Massive API allows 5 requests per minute
        self.edgarLimiter = RateLimiter("edgar", 5, 1)              # No official limit, 5 reqs per second will be safe
        self.fredLimiter = RateLimiter("fred", 120, 60)             # FRED allows 120 requests per minute
        self.finraLimiter = RateLimiter("finracdn", 300, 60)        # FINRA's CDN doesnt have a limit, 300/min should be safe
        
        self.alpacaNewsDlLimiter = RateLimiter("alpacaNewsDl", 200, 60)   # Naughty! Im using 2 api keys for Alpaca