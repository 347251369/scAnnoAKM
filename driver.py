import torch
import os
import re
import traceback
import numpy as np
import random
from dataset import *
from DPA import *
from Brain import *
from QVA import *
from KRA import *
from scAnnoModel import *
os.environ.setdefault("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
os.environ.setdefault("QWEN_API_KEY", "***********")
os.environ.setdefault("BRAIN_MODEL", "qwen2.5-72b-instruct")
os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
os.environ["no_proxy"]  = "127.0.0.1,localhost,::1"
for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(k, None)
brain_client = LLMClient(os.getenv("QWEN_API_BASE"), os.getenv("QWEN_API_KEY"), os.getenv("BRAIN_MODEL"))


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class train_Config:
    def __init__(self, file_addr, random_seed):
        self.is_train = True
        self.is_predict = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = random_seed

        self.pretrain_epochs = 20
        self.pretrain_lr = 1e-3
        self.pretrain_batch_size = 32
        self.pretrain_model_addr = f'results/{file_addr}/pretrain_model_seed{random_seed}.pth'
        self.gene_info_addr = f'results/{file_addr}/gene_info_seed{random_seed}.pth'

        self.train_epochs = 3
        self.train_lr = 1e-4
        self.train_batch_size = 8
        self.seq_len = 2000
        self.marker_len = 50
        self.train_model_addr = f'results/{file_addr}/train_model_seed{random_seed}.pth'

        self.predict_batch_size = 32
        self.weight_decay = 5e-3
        self.gradient_clip_norm = 2.0
        self.delta = 0.2
        self.fixed_view_weights = (0.2, 0.2, 0.6)
        self.results_addr = f'results/{file_addr}'
        self.score_dump_addr = f'results/{file_addr}/score_dump_seed{random_seed}.npz'


def run(addr,seed):
    random_seed=seed
    set_seed(random_seed)

    # load dataset
    file_addr = addr
    dataset= load_dataset(file_addr)
    print(f"load:{dataset.n_obs} cells * {dataset.n_vars} genes")

    config = train_Config(file_addr, random_seed)

    # split dataset
    reference_dataset, target_dataset = split_dataset(file_addr, dataset, test_size=0.3, random_seed=random_seed)
    print("Split dataset successfully")
    print(f"Reference dataset: {reference_dataset.n_obs} cells * {reference_dataset.n_vars} genes")
    print(f"Target dataset: {target_dataset.n_obs} cells * {target_dataset.n_vars} genes")
    
    #load DPA
    global_norm_factors = get_global_normalization_factors(dataset.copy())
    ref_seq, ref_label, tar_seq, tar_label = DPA(brain_client, file_addr, reference_dataset, target_dataset,  global_norm_factors)
    print("Use DPA to get sequences successfully")

    #load knowledge_acquire
    label_total, descriptions, markers = knowledge_acquire(file_addr)
    print("Get knowledges successfully")

    #load KRA
    cell_knowledges = KRA(brain_client, file_addr, label_total, descriptions, markers, config)
    print("Use KRA to generate multi-views knowledges successfully")

    #load QVA
    cell_knowledges = QVA(brain_client, file_addr, descriptions, cell_knowledges)
    print(f"Use QVA to update the cell texts successfully")

    gene_knowledges = load_gene_knowledge(file_addr)
    print(f"Load gene knowledge: {len(gene_knowledges)} genes")

    run_scAnnoModel(file_addr, gene_knowledges, cell_knowledges, ref_seq, ref_label, tar_seq, tar_label, config)

if __name__ == "__main__":
    addr = 'retina'
    seed = 42
    run(addr, seed)
