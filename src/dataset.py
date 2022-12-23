import torch


class TweetDataset:
    def __init__(self, tweets, targets):
        self.tweets = tweets
        self.targets = targets

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        tweet = self.tweets[item, :]
        target = self.targets[item]
        return {"tweet": torch.tensor(tweet, dtype=torch.long), "target": torch.tensor(target, dtype=torch.float)}
