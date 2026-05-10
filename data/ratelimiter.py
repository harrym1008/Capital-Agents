import time


class RateLimiter:
    def __init__(self, name, maxCalls, period):
        self.name = name
        self.maxCalls = maxCalls
        self.period = period
        self.callInterval = period / maxCalls
        self.lastCallTime = 0
    

    def wait(self):
        currentTime = time.time()
        
        if self.lastCallTime == 0:
            self.lastCallTime = currentTime
            return
        
        elapsedTime = currentTime - self.lastCallTime
        if elapsedTime < self.callInterval:
            timeToWait = self.callInterval - elapsedTime
            time.sleep(timeToWait)

        time.sleep(0.05)
        self.lastCallTime = time.time()

        
    def got429(self, iters=0):
        waitTime = self.period * iters/4 + 1
        if iters == 1:
            waitTime = self.period * 4

        print(f"[{self.name}] Received 429 \"Too Many Requests\". Waiting for {waitTime} seconds...")
        time.sleep(waitTime)   # Exponential backoff


    def non429Error(self, e):
        print(f"[{self.name}] Received non-429 error - {e.__class__.__name__}: {e}. Waiting for 15 seconds before retrying...")
        time.sleep(15)
