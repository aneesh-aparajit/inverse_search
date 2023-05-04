from torchview import draw_graph
from src.models.bert import BertModel
import torch
import yaml

with open('../config/config_v1.yaml') as f:
    config = yaml.safe_load(f)


def visualize_bert():
    inputs = torch.randint(
        low=0,
        high=config["LANGUAGE"]["BASELINE_CONFIG"]["NUM_EMBEDDINGS"],
        size=(32, 128),
    )

    model_graph = draw_graph(
        BertModel(config=config), input_data=inputs,
        graph_name='BertModel', 
        expand_nested=True, save_graph=True, filename="../../reports/images/bert/BertModel"
    )

    model_graph.visual_graph
