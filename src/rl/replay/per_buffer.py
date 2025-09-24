from segment_tree import SumSegmentTree
import numpy as np, random, redis, pickle


class PrioritizedReplay:
    α = 0.6
    β = 0.4
    β_inc = 1e-5

    def __init__(self, key: str, capacity=1_000_000):
        self.key, self.capacity = key, capacity
        self.r = redis.Redis()
        self.tree = SumSegmentTree(capacity)
        self.max_p = 1.0

    def add(self, transition, priority=None):
        p = priority or self.max_p
        idx = int(self.r.incr(f"{self.key}:ptr") % self.capacity)
        self.tree[idx] = p**self.α
        self.r.set(f"{self.key}:{idx}", pickle.dumps(transition))
        self.max_p = max(self.max_p, p)

    def sample(self, batch_size=256):
        seg = self.tree.total() / batch_size
        idxs, probs, batch = [], [], []
        for i in range(batch_size):
            a, b = seg * i, seg * (i + 1)
            idx = self.tree.find_prefixsum_idx(random.uniform(a, b))
            p = self.tree[idx]
            trans = pickle.loads(self.r.get(f"{self.key}:{idx}"))
            idxs.append(idx)
            probs.append(p)
            batch.append(trans)
        probs = np.array(probs) / self.tree.total()
        weights = (len(self) * probs) ** (-self.β)
        self.β = min(1.0, self.β + self.β_inc)
        return batch, idxs, weights / weights.max()

    def update_priorities(self, idxs, priorities):
        for i, p in zip(idxs, priorities):
            self.tree[i] = p**self.α

    def __len__(self):
        return min(int(self.r.get(f"{self.key}:ptr") or 0), self.capacity)
