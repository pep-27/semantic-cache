import time

class Metrics:
    def __init__(self):
        self.records = []

    def record_call(self, hit: bool, latency: float):
        """
        record every call (hit or not + latency)
        """
        self.records.append({
            "hit": hit,
            "latency": latency
        })

    def summary(self):
        """
        calculate the average metrics
        """
        total_calls = len(self.records)
        total_hits = sum(r["hit"] for r in self.records)
        avg_latency = sum(r["latency"] for r in self.records) / total_calls if total_calls > 0 else 0

        hit_rate = total_hits / total_calls if total_calls > 0 else 0
        return {
            "total_calls": total_calls,
            "total_hits": total_hits,
            "hit_rate": round(hit_rate, 3),
            "avg_latency": round(avg_latency, 3)
        }


# ✅ test example
if __name__ == "__main__":
    m = Metrics()

    for i in range(5):
        start = time.time()
        time.sleep(0.1 * i)  # simulate different latency
        m.record_call(hit=(i % 2 == 0), latency=time.time() - start)

    print(m.summary())
