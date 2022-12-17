import torch
import torch.nn as nn


def train(data_loader, model, optimizer, device):

    # set the model to training mode
    for data in data_loader:
        # fetch tweet and target from the dict
        tweets = data["tweet"]
        targets = data["target"]

        # move the data to device that we want to use
        if device:
            tweets = tweets.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

        # clear the gradients
        optimizer.zero_grad()

        # make predictions from the model
        predictions = model(tweets)

        # calculate the loss
        loss = nn.BCEWithLogitsLoss()(
            predictions,
            targets.view(-1, 1)
        )

        # compute gradients of loss w.r.t.
        # all parameters of the model that are trainable
        loss.backward()

        # single optimization step
        optimizer.step()


def evaluate(data_loader, model, device):
    # initialize empty list to store predictions and targets
    final_predictions = []
    final_targets = []

    # put the model in eval mode
    model.eval()

    # disable gradient calculation
    with torch.no_grad():
        for data in data_loader:
            tweets = data["tweet"]
            targets = data["target"]
            if device:
                tweets = tweets.to(device, dtype=torch.long)
            # targets = targets.to(device, dtype=torch.float)

            # make predictions
            predictions = model(tweets)

            # move predictions and targets to list
            # we need to move predictions and targets to cpu too
            predictions = predictions.cpu().numpy().tolist()
            # targets = data["target"].cpu().numpy().tolist()
            final_predictions.extend(predictions)
            final_targets.extend(targets)

        # return final predictions and targets
        return final_predictions, final_targets
