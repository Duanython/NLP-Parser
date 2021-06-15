import torch
from tqdm import tqdm
import math
from dyfparser.utils.toolkit import AverageMeter
from dyfparser.utils.minibatches import minibatches


def train(parser, train_data, dev_data, output_path, batch_size=1024, n_epochs=10, lr=0.0005):
    best_dev_UAS = 0

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(parser.model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
        dev_UAS = train_for_epoch(parser, train_data, dev_data, optimizer, loss_func, batch_size)
        if dev_UAS > best_dev_UAS:
            best_dev_UAS = dev_UAS
            print("New best dev UAS! Saving model.")
            torch.save(parser.model.state_dict(), output_path)
        print("")


def train_for_epoch(parser, train_data, dev_data, optimizer, loss_func, batch_size):
    parser.model.train()  # Places model in "train" mode, i.e. apply dropout layer
    n_minibatches = math.ceil(len(train_data) / batch_size)
    loss_meter = AverageMeter()

    with tqdm(total=n_minibatches) as prog:
        for i, (train_x, train_y) in enumerate(minibatches(train_data, batch_size)):
            # train_x (batch_size,36) train_y (batch_size,3) 均为ndarray
            optimizer.zero_grad()  # remove any baggage in the optimizer
            # train_x (batch_size,36) train_y (batch_size) 均为tensor
            train_x = torch.from_numpy(train_x).long()
            train_y = torch.from_numpy(train_y.nonzero()[1]).long()
            logits = parser.model.forward(train_x)
            loss = loss_func(logits, train_y)
            loss.backward()
            optimizer.step()
            prog.update(1)
            loss_meter.update(loss.item())

    print("Average Train Loss: {}".format(loss_meter.avg))

    print("Evaluating on dev set", )
    parser.model.eval()  # Places model in "eval" mode, i.e. don't apply dropout layer
    dev_UAS, _ = parser.parse(dev_data)
    print("- dev UAS: {:.2f}".format(dev_UAS * 100.0))
    return dev_UAS
