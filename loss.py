import random
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Ranking_Loss(nn.Module):
    def __init__(self, N, minibatch):
        super().__init__()
        self.N = N
        self.minibatch = minibatch

    def calculate_l(self, x1, y1, x2, y2, gt):

        pix1 = gt[:, x1, y1]
        pix2 = gt[:, x2, y2]

        ls = torch.zeros(self.minibatch)

        for i in range(self.minibatch):
            if pix1[i] / (pix2[i] + 1e-7) > 1.02:
                ls[i] = 1
            elif pix2[i] / (pix1[i] + 1e-7) > 1.02:
                ls[i] = -1
            
        return ls

    def calculate_phi(self, x1, y1, x2, y2, gt, output):
        ls = self.calculate_l(x1, y1, x2, y2, gt).to(device)
        pred_depth = (output[:, x1, y1] - output[:, x2, y2]).to(device)
        log_loss = torch.mean(torch.log(1 + torch.exp(-ls[ls != 0] * pred_depth[ls != 0])))
        
        if pred_depth[ls==0].shape[0] != 0:
            squared_loss = torch.mean(pred_depth[ls == 0] ** 2) 
            return log_loss + squared_loss
        
        return log_loss

    def random_sampling(self):
        x1 = random.randint(0, 383)
        y1 = random.randint(0, 383)
        x2 = random.randint(0, 383)
        y2 = random.randint(0, 383)

        return x1, y1, x2, y2

    def calculate_main(self, gt, output):
        gt = gt.squeeze(1)
        output = output.squeeze(1)
        loss = 0
        for i in range(self.N):
            x1, y1, x2, y2 = self.random_sampling()
            loss += self.calculate_phi(x1, y1, x2, y2, gt, output)

        return loss / self.N