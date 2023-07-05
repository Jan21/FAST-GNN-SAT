
import torch

# save model
def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def test_model(model, dataset):
    model.eval()
    test = dataset
    acc = 0
    for data in test:
        #x = data.x
        #edge_index = data.edge_index
        y = data.y
        y_hat = model(data)
        pred =  y_hat.max(dim=1)[1]
        acc += pred.eq(y).sum()
    acc = acc.item() / len(test)
    return acc

def save_results(results, path):
    with open(path, 'w') as f:
        for item in results:
            f.write("%s " % item)
            f .write("\n")


