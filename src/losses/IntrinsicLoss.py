import torch

class IntrinsicLoss(torch.nn.Module):
    def __init__(self):
        super(IntrinsicLoss, self).__init__()

    def forward(self, X, W):
        '''
        Computes the loss
        :param X: Network logits bs x 6
        :param W: Ground truth eigenvector  bs x 4
        :return: Loss vector bs x 1
        '''

        a11 = X[:, 0]*X[:, 2]+X[:, 1]*X[:, 3]
        a12 = X[:, 0]+X[:, 2]
        a13 = X[:, 1]+X[:, 3]
        a14 = torch.ones(X.shape[0])
        a21 = X[:, 0]*X[:, 4]+X[:, 1]*X[:, 5]
        a22 = X[:, 0]+X[:, 4]
        a23 = X[:, 1]+X[:, 5]
        a24 = torch.ones(X.shape[0])
        a31 = X[:, 2]*X[:, 4]+X[:, 3]*X[:, 5]
        a32 = X[:, 2]+X[:, 4]
        a33 = X[:, 3]+X[:, 5]
        a34 = torch.ones(X.shape[0])

        a1 = torch.stack([a11, a12, a13, a14])
        a2 = torch.stack([a21, a22, a23, a24])
        a3 = torch.stack([a31, a32, a33, a34])

        A = torch.stack([a1, a2, a3])
        A = A.transpose(1, 2).transpose(0, 1)

        return torch.matmul(W.transpose(1, 2), torch.matmul(A.transpose(1, 2), torch.matmul(A, W))).squeeze(2).mean()


if __name__ == '__main__':
    x = torch.randn(10, 6, dtype=torch.float, requires_grad=True)
    w = torch.randn(10, 4, 1, dtype=torch.float, requires_grad=True)
    print('x:', x.shape)
    print('w', w.shape)
    criterion = IntrinsicLoss()
    loss = criterion.forward(x, w)
    print('loss:', loss)
    loss.backward()
    print('x:', x)
    print('w', w)
