import torch

class PolarLoss(torch.nn.Module):
    def __init__(self, radius):
        super(PolarLoss, self).__init__()
        self.radius = radius

    def forward(self, X, Y):
        '''
        Computes the polar distance between the two points
        :param X:
        :param Y:
        :return:
        '''
        Rx = torch.sqrt(torch.sum(X.reshape(3, 2) * X.reshape(3, 2), dim=1))
        Ry = torch.sqrt(torch.sum(Y.reshape(3, 2) * Y.reshape(3, 2), dim=1))

        Tx = torch.atan2(X.reshape(3, 2)[:, 0], X.reshape(3, 2)[:, 1])
        Ty = torch.atan2(Y.reshape(3, 2)[:, 0], Y.reshape(3, 2)[:, 1])

        pointwise_loss = torch.sqrt(torch.pow(Ry - Rx, 2) + 2 * self.radius * torch.pow(Ty - Tx, 2))

        print(Rx, Ry)
        print(Tx, Ty)
        print(pointwise_loss)

        return torch.sum(pointwise_loss, dim=0)


if __name__ == '__main__':
        criterion = PolarLoss(432.)
        X = torch.tensor([-385, -2, -385, -2, -385, -2], dtype=torch.float, requires_grad=True)
        Y = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float, requires_grad=True)
        loss = criterion(X, Y)
        print(loss)
        loss.backward()
        print(X, Y)

