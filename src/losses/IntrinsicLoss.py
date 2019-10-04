import torch

class IntrinsicLoss(torch.nn.Module):
    def __init__(self):
        super(IntrinsicLoss, self).__init__()

    def forward(self, X, W):
        losses = list()
        for x, w in zip(X, W):
            A = torch.tensor([[x[0]*x[2]+x[1]*x[3], x[0]+x[2], x[1]+x[3], 1],
                              [x[0]*x[4]+x[1]*x[5], x[0]+x[4], x[1]+x[5], 1],
                              [x[2]*x[4]+x[3]*x[5], x[2]+x[4], x[3]+x[5], 1]],
                             dtype=torch.float, requires_grad=True)

            losses.append(w.t() @ A.t() @ A @ w)
        return torch.tensor(losses)

if __name__ == '__main__':
    x = torch.randn(10, 6, 1, dtype=torch.float, requires_grad=True)
    w = torch.randn(10, 4, 1, dtype=torch.float, requires_grad=True)
    print('x:', x)
    print('w', w)
    criterion = IntrinsicLoss()
    loss = criterion.forward(x, w)
    print('loss:', loss)
    loss.backward()
    print('x:', x)
    print('w', w)
