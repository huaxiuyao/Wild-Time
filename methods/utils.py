import ipdb
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from .lisa import lisa
from .mixup import mixup_data, mixup_criterion

def prepare_data(x, y, dataset_name: str):
    if dataset_name == 'drug':
        x[0] = x[0].cuda()
        x[1] = x[1].cuda()
        y = y.cuda()
    elif dataset_name == 'precipitation':
        x['categorical'] = x['categorical'].cuda()
        x['continuous'] = x['continuous'].cuda()
        y = y.cuda()
    elif dataset_name == 'mimic':
        y = torch.cat(y).type(torch.LongTensor).cuda()
    elif dataset_name in ['arxiv', 'huffpost']:
        x = x.to(dtype=torch.int64).cuda()
        if len(y.shape) > 1:
            y = y.squeeze(1).cuda()
    elif dataset_name in ['fmow', 'yearbook']:
        x = x.cuda()
        if len(y.shape) > 1:
            y = y.squeeze(1).cuda()
    return x, y


def forward_pass(x, y, dataset, network, criterion, use_lisa: bool, use_mixup: bool, cut_mix: bool, mix_alpha=2.0):
    if use_lisa:
        if str(dataset) in ['arxiv', 'huffpost']:
            x = network.model[0](x)
            sel_x, sel_y = lisa(x, y, dataset=dataset, mix_alpha=mix_alpha,
                                num_classes=dataset.num_classes, time_idx=dataset.current_time,
                                cut_mix=cut_mix, embedding=network.model[0])
            logits = network.model[1](sel_x)
        elif str(dataset) in ['precipitation']:
            x = network.net(x)
            sel_x, sel_y = lisa(x, y, dataset=dataset, mix_alpha=mix_alpha,
                                num_classes=dataset.num_classes, time_idx=dataset.current_time,
                                cut_mix=cut_mix, embedding=network.net)
            logits = network.net.linear_layers(sel_x)
            logits = network.logits(logits)
        elif str(dataset) in ['drug']:
            sel_x0, sel_y = lisa(x[0], y, dataset=dataset, mix_alpha=mix_alpha,
                                num_classes=dataset.num_classes, time_idx=dataset.current_time,
                                cut_mix=cut_mix)
            sel_x1, sel_y = lisa(x[1], y, dataset=dataset, mix_alpha=mix_alpha,
                                num_classes=dataset.num_classes, time_idx=dataset.current_time,
                                cut_mix=cut_mix)
            sel_x = [sel_x0, sel_x1]
            logits = network(sel_x)
        elif str(dataset) in ['mimic']:
            x = network.get_cls_embed(x)
            sel_x, sel_y = lisa(x, y, dataset=dataset, mix_alpha=mix_alpha,
                                num_classes=dataset.num_classes, time_idx=dataset.current_time,
                                cut_mix=cut_mix, embedding=network.get_cls_embed)
            logits = network.fc(sel_x)
        else:
            sel_x, sel_y = lisa(x, y, dataset=dataset, mix_alpha=mix_alpha,
                                num_classes=dataset.num_classes, time_idx=dataset.current_time,
                                cut_mix=cut_mix)
            logits = network(sel_x)
        y = torch.argmax(sel_y, dim=1)
        loss = criterion(logits, y)

    elif use_mixup:
        if str(dataset) in ['arxiv', 'huffpost']:
            x = network.model[0](x)
            x, y_a, y_b, lam = mixup_data(x, y, mix_alpha=mix_alpha)
            logits = network.model[1](x)
        elif str(dataset) in ['mimic']:
            x = network.get_cls_embed(x)
            x, y_a, y_b, lam = mixup_data(x, y, mix_alpha=mix_alpha)
            logits = network.fc(x)
        elif str(dataset) in ['drug']:
            x0, y_a, y_b, lam = mixup_data(x[0], y, mix_alpha=mix_alpha)
            x1, y_a, y_b, lam = mixup_data(x[1], y, mix_alpha=mix_alpha)
            x = [x0, x1]
            y_a = y_a.float()
            y_b = y_b.float()
            logits = network(x).squeeze(1).float()
        elif str(dataset) in ['precipitation']:
            x = network.net(x)
            x, y_a, y_b, lam = mixup_data(x, y, mix_alpha=mix_alpha)
            y_a = y_a.squeeze(1)
            y_b = y_b.squeeze(1)
            logits = network.net.linear_layers(x)
            logits = network.logits(logits)
        else:
            x, y_a, y_b, lam = mixup_data(x, y, mix_alpha=mix_alpha)
            x, y_a, y_b = map(Variable, (x, y_a, y_b))
            logits = network(x)
        loss = mixup_criterion(criterion, logits, y_a, y_b, lam)

    else:
        logits = network(x)
        if str(dataset) in ['drug']:
            logits = logits.squeeze().double()
            y = y.squeeze().double()
        elif str(dataset) in ['arxiv', 'fmow', 'huffpost', 'precipitation', 'yearbook']:
            if len(y.shape) > 1:
                y = y.squeeze(1)
            if str(dataset) in ['precipitation'] and type(criterion) == torch.nn.modules.loss.MSELoss:
                logits = logits.squeeze(1)
                y = y.float()
        loss = criterion(logits, y)

    return loss, logits, y


def split_into_groups(g):
    """
    From https://github.com/p-lambda/wilds/blob/f384c21c67ee58ab527d8868f6197e67c24764d4/wilds/common/utils.py#L40.
    Args:
        - g (Tensor): Vector of groups
    Returns:
        - groups (Tensor): Unique groups present in g
        - group_indices (list): List of Tensors, where the i-th tensor is the indices of the
                                elements of g that equal groups[i].
                                Has the same length as len(groups).
        - unique_counts (Tensor): Counts of each element in groups.
                                 Has the same length as len(groups).
    """
    unique_groups, unique_counts = torch.unique(g, sorted=False, return_counts=True)
    group_indices = []
    for group in unique_groups:
        group_indices.append(
            torch.nonzero(g == group, as_tuple=True)[0])
    return unique_groups, group_indices, unique_counts


def plot_accuracy(
        title: str,
        tasks,
        accuracies,
        save_path: str
):
    plt.plot(tasks, accuracies)
    plt.xlabel("Task")
    plt.ylabel("Accuracy")
    plt.ylim([0.0, 1.0])
    plt.title(title)
    plt.savefig(save_path)