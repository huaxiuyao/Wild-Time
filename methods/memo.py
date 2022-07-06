import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from methods.transformations import aug
from torch.autograd import Variable

def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits

def adapt_single(model, image, optimizer, niter, batch_size):
    model.eval()
    for iter in range(niter):
        inputs = [aug(image) for _ in range(batch_size)]
        inputs = torch.stack(inputs).cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss, logits = marginal_entropy(outputs)
        loss.backward()
        optimizer.step()