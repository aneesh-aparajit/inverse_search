from torchview import draw_graph
from src.models.bert import BertModel
from src.models.vit import ViTModel
from src.models.clip import ClipModel
import torch
import yaml


def visualize_bert():
    with open('../config/cfg1.yaml') as f:
        config = yaml.safe_load(f)
    inputs = torch.randint(
        low=0,
        high=config["BERT"]["BASELINE_CONFIG"]["NUM_EMBEDDINGS"],
        size=(32, 128),
    )

    model_graph = draw_graph(
        BertModel(config=config), input_data=inputs,
        graph_name='BertModel', 
        expand_nested=True, save_graph=True, filename="../../reports/images/bert/BertModel"
    )

    model_graph.visual_graph

def visualize_vit():
    with open('../config/cfg1.yaml') as f:
        config = yaml.safe_load(f)
    
    x = torch.randn(size=(32, 3, 224, 224))

    model_graph = draw_graph(
        ViTModel(config=config), input_data=x,
        graph_name='ViTModel', 
        expand_nested=True, save_graph=True, filename="../../reports/images/vit/ViTModel"
    )

    model_graph.visual_graph

def visualize_clip():
    with open('../config/cfg1.yaml') as f:
        config = yaml.safe_load(f)
    
    x = torch.randn(size=(32, 3, 224, 224))
    inputs = torch.randint(
        low=0,
        high=config["BERT"]["BASELINE_CONFIG"]["NUM_EMBEDDINGS"],
        size=(32, 128),
    )

    model_graph = draw_graph(
        ClipModel(cnn=ViTModel(config=config), lm=BertModel(config=config)), 
        input_data=({
            'input_ids': x, 
            'image': inputs
        }),
        graph_name='ClipModel', 
        expand_nested=True, save_graph=True, filename="../../reports/images/clip/ClipModel"
    )

    model_graph.visual_graph

if __name__ == '__main__':
    visualize_vit()
