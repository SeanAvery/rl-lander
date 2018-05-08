from collections import deque
import random

class Memory():
    def __init__(self, max_size):
        self.memories = deque()
        self.max_size = max_size

    is_full = lambda x: len(self.memories) > self.max_size

    def append(self, item):
        if is_full:
            self.memories.popleft()
            self.memories.append(item)
        else:
            self.memories.append(item)

    def sample(self, num_items):
        return random.sample(self.memories, min(len(self.memory), num_items))
