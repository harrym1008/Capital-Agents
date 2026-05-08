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
        waitTime = self.period * iters/5
        print(f"[{self.name}] Received 429 \"Too Many Requests\". Waiting for {waitTime} seconds...")
        time.sleep(waitTime)   # Exponential backoff

