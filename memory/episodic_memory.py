import collections

class EpisodicMemory:
    def __init__(self, capacity=1000):
        self.memory = collections.deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })

    def get_recent(self, n=10):
        return list(self.memory)[-n:]

    def clear(self):
        self.memory.clear()
