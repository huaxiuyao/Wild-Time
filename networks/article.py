import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertForSequenceClassification


class DistilBertClassifier(DistilBertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        outputs = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        return outputs


class DistilBertFeaturizer(DistilBertModel):
    def __init__(self, config):
        super().__init__(config)
        self.d_out = config.hidden_size

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        hidden_state = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        pooled_output = hidden_state[:, 0]
        return pooled_output


class ArticleNetwork(nn.Module):
    def __init__(self, num_classes):
        super(ArticleNetwork, self).__init__()
        featurizer = DistilBertFeaturizer.from_pretrained("distilbert-base-uncased")
        classifier = nn.Linear(featurizer.d_out, num_classes)
        self.model = nn.Sequential(featurizer, classifier)

    def forward(self, x):
        return self.model(x)