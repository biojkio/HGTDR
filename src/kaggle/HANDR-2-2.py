# %% [markdown]
# 有元路径权重统计和绘图

# %%
from torch_geometric.nn import HANConv, Linear
from torch_geometric.loader import HGTLoader
from torch_geometric.data import HeteroData
from torch_geometric.transforms import AddMetaPaths
import torch.nn.functional as F
import pickle
import torch.nn as nn
import pandas as pd
# from utils import *
import random
import torch
import copy
import os
import sys
from tqdm import tqdm

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
node_type1 = 'drug'
node_type2 = 'disease'
rel = 'indication'

# 预设元路径
# 预设元路径
METAPATHS = [
    # 1. Drug → Disease → Disease → Drug
    [('drug', 'indication', 'disease'), ('disease', 'disease_disease', 'disease'), ('disease', 'indication', 'drug')],
    # 2. Drug → Disease → Drug
    [('drug', 'indication', 'disease'), ('disease', 'indication', 'drug')],
    # 3. Drug → Protein → Disease → Drug
    [('drug', 'drug_protein', 'gene/protein'), ('gene/protein', 'disease_protein', 'disease'), ('disease', 'indication', 'drug')],
    # 4. Drug → Protein → Drug
    [('drug', 'drug_protein', 'gene/protein'), ('gene/protein', 'drug_protein', 'drug')],
    # 5. Disease → Disease → Disease
    [('disease', 'disease_disease', 'disease'), ('disease', 'disease_disease', 'disease')],
    # 6. Drug → Phenotype → Drug
    [('drug', 'drug_effect', 'effect/phenotype'), ('effect/phenotype', 'drug_effect', 'drug')],
    # 7. Disease → Phenotype → Disease
    [('disease', 'disease_phenotype_positive', 'effect/phenotype'), ('effect/phenotype', 'disease_phenotype_positive', 'disease')],
    # 8. Drug → Protein → Phenotype → Disease → Drug
    [('drug', 'drug_protein', 'gene/protein'), ('gene/protein', 'phenotype_protein', 'effect/phenotype'), ('effect/phenotype', 'disease_phenotype_positive', 'disease'), ('disease', 'indication', 'drug')],
]


# %%
config = {
    "num_samples": 512,
    "batch_size": 164,
    "dropout": 0.5,
    "epochs": 300,
    "file_name": "HANDR-2",
    # 早停参数
    "early_stopping_patience": 20,      # 容忍验证损失不改善的最大 epoch 数
    "early_stopping_min_delta": 1e-4,   # 认为"有改善"所需的最小降幅
    "early_stopping_smooth_window": 3   # 平滑窗口大小（消除随机负采样噪声）
}

# %% [markdown]
# # Load data

# %%
primekg_file = '/kaggle/input/datasets/jkiobio/hgtdr-data/data/kg.csv'
df = pd.read_csv(primekg_file, sep =",")

# %% [markdown]
# ### Get drugs and diseases which are used in indication relation.

# %% [markdown]
# ### Remove drug and disease nodes that do not contribute to at least one indication edge. 

# %%
# 确保每条边连接的是 drug 和 disease（顺序不限）
valid_rows = (
    ((df['x_type'] == 'drug') & (df['y_type'] == 'disease')) |
    ((df['x_type'] == 'disease') & (df['y_type'] == 'drug'))
)
drug_disease_pairs = df[(df['relation'] == 'indication') & valid_rows]

# 提取所有 x 和 y 的 (type, index) 对
x_mask = drug_disease_pairs['x_type'].isin([node_type1, node_type2])
y_mask = drug_disease_pairs['y_type'].isin([node_type1, node_type2])

# 合并所有有效实体
all_entities = pd.concat([
    drug_disease_pairs.loc[x_mask, ['x_type', 'x_index']].rename(columns={'x_type': 'type', 'x_index': 'index'}),
    drug_disease_pairs.loc[y_mask, ['y_type', 'y_index']].rename(columns={'y_type': 'type', 'y_index': 'index'})
])

# 分别提取 drug 和 disease
drugs = all_entities[all_entities['type'] == node_type1]['index'].unique().tolist()
diseases = all_entities[all_entities['type'] == node_type2]['index'].unique().tolist()

# 确保是 set 以加速
valid_drugs = set(drugs)
valid_diseases = set(diseases)

# 定义检查函数 (向量化操作的核心是避免 apply，但这里逻辑稍复杂，用布尔掩码最清晰)
# 检查 x 节点是否有效
check_x = (
    (~df['x_type'].isin(['drug', 'disease'])) |  # 情况1: 不是目标类型 -> 有效
    ((df['x_type'] == 'drug') & df['x_index'].isin(valid_drugs)) |      # 情况2: 是drug且在列表 -> 有效
    ((df['x_type'] == 'disease') & df['x_index'].isin(valid_diseases))  # 情况3: 是disease且在列表 -> 有效
)

# 检查 y 节点是否有效
check_y = (
    (~df['y_type'].isin(['drug', 'disease'])) |
    ((df['y_type'] == 'drug') & df['y_index'].isin(valid_drugs)) |
    ((df['y_type'] == 'disease') & df['y_index'].isin(valid_diseases))
)

# 只有 x 和 y 同时有效，才保留
df_cleaned = df[check_x & check_y].reset_index(drop=True)

# 1. 构建格式化后的列 (使用 f-string 或 vectorized string 操作)
# 注意：确保 index 列是字符串类型，防止数字和字符串拼接报错
head_nodes = df_cleaned['x_type'] + '::' + df_cleaned['x_index'].astype(str)
tail_nodes = df_cleaned['y_type'] + '::' + df_cleaned['y_index'].astype(str)

# 2. 组装新的 DataFrame
new_df = pd.DataFrame({
    0: head_nodes,
    1: df_cleaned['relation'],
    2: tail_nodes
})

# 3. 去重并转换为列表
# drop_duplicates() 会移除完全相同的行 (头-关系-尾 都相同)
df = new_df.drop_duplicates()
triplets = df.values.tolist()

# 打印预览
print(f"生成三元组数量: {len(triplets)}")
print(f"示例数据: {triplets[:3]}")

# %%
entity_dictionary = {}

for src, _, dest in triplets:
    for node in [src, dest]:
        n_type, n_id = node.split('::', 1)
        
        # setdefault: 如果 key 不存在，初始化为空字典，并返回该字典
        type_dict = entity_dictionary.setdefault(n_type, {})
        
        # 如果实体不在字典中，赋予新 ID (当前长度)
        if node not in type_dict:
            type_dict[node] = len(type_dict)
            


# %%
from collections import defaultdict

# 使用 defaultdict 自动初始化列表，避免 if-else 判断
edge_dictionary = defaultdict(list)

for src, relation, dest in triplets:
    # 1. 解析类型 (只取 '::' 之前的部分)
    src_type = src.split('::', 1)[0]
    dest_type = dest.split('::', 1)[0]
    
    # 2. 获取整数 ID (直接从之前构建的 entity_dictionary 中查找)
    src_int_id = entity_dictionary[src_type][src]
    dest_int_id = entity_dictionary[dest_type][dest]
    
    # 3. 构建边类型键 (SrcType, Relation, DstType)
    etype = (src_type, relation, dest_type)
    
    # 4. 添加边 (defaultdict 会自动处理列表初始化)
    edge_dictionary[etype].append((src_int_id, dest_int_id))

# 如果需要转回普通字典 (可选，通常 defaultdict 也能直接用于后续处理)
edge_dictionary = dict(edge_dictionary)


# %% [markdown]
# 节点类型组成维度 
# DrugNode2Vec + ChemBERTa 128 + 767 = 895 
# Non-DrugNode2Vec + PubMedBERT 128 + 768 = 896

# %%
# Cell 18+20: 初始化 HeteroData 并填充嵌入
import numpy as np

# 已删除 NODE2VEC_DIM 常量

CHEMBERTA_DIM  = 767
PUBMEDBERT_DIM = 768

# 加载嵌入文件
# 已删除 node2vec_df 的加载
pubmedbert_df = pd.read_pickle('/kaggle/input/datasets/jkiobio/hgtdr-data/data/pubmedbert_embeddings.pkl')
smiles_df     = pd.read_pickle('/kaggle/input/datasets/jkiobio/hgtdr-data/data/smiles_embeddings.pkl')

# 创建字典
# 已删除 node2vec_dict 的创建
pubmedbert_dict = dict(zip(pubmedbert_df['id'], pubmedbert_df['embedding']))
smiles_dict     = dict(zip(smiles_df['id'],     smiles_df['embedding']))

# 初始化节点特征
data = HeteroData()
for key in entity_dictionary.keys():
    num_nodes = len(entity_dictionary[key])
    # 修改维度：移除 NODE2VEC_DIM
    dim = CHEMBERTA_DIM if key == 'drug' else PUBMEDBERT_DIM
    data[key].x  = torch.zeros((num_nodes, dim))
    data[key].id = torch.arange(num_nodes)

# 添加边
for key in edge_dictionary:
    data[key].edge_index = torch.transpose(
        torch.IntTensor(edge_dictionary[key]), 0, 1
    ).long().contiguous()

# 填充嵌入
for node_type, mapping in tqdm(entity_dictionary.items(), desc='填充节点嵌入'):
    for entity_id, hgt_id in mapping.items():

        # 已删除 node2vec 嵌入的填充逻辑

        if node_type == 'drug':
            if entity_id in smiles_dict:
                data[node_type].x[hgt_id, :] = torch.tensor( # 修改索引
                    np.array(smiles_dict[entity_id], dtype=np.float32)
                )
        else:
            if entity_id in pubmedbert_dict:
                data[node_type].x[hgt_id, :] = torch.tensor( # 修改索引
                    np.array(pubmedbert_dict[entity_id], dtype=np.float32)
                )

print('节点维度:')
for k in data.node_types:
    print(f'  {k}: {data[k].x.shape}')

# %% [markdown]
# ### Load train and validation data of one fold.

# %%
file = open('/kaggle/input/datasets/jkiobio/cv-mp-data/CV_mp_data/train_mp1.pkl', 'rb')
train_data = pickle.load(file)

# %%
file = open('/kaggle/input/datasets/jkiobio/cv-mp-data/CV_mp_data/val_mp1.pkl', 'rb')
val_data = pickle.load(file)

# %% [markdown]
# ### Creating mask.

# %%
# 2. 创建 Mask
# 获取边数 (逻辑与原代码完全一致)
drug_disease_num = train_data[(node_type1, rel, node_type2)]['edge_index'].shape[1]

# 随机采样 80% 的索引
mask = random.sample(range(drug_disease_num), int(drug_disease_num * 0.8))

# 初始化正向边 mask 并赋值
train_data[(node_type1, rel, node_type2)]['mask'] = torch.zeros(drug_disease_num, dtype=torch.bool)
train_data[(node_type1, rel, node_type2)]['mask'][mask] = True

# 初始化反向边 mask 并赋值 (逻辑与原代码完全一致，保持显式写出以便阅读)
train_data[(node_type2, rel, node_type1)]['mask'] = torch.zeros(drug_disease_num, dtype=torch.bool)
train_data[(node_type2, rel, node_type1)]['mask'][mask] = True

# %% [markdown]
# ### Define model.

# %%
class HAN(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, dropout, metadata):
        super().__init__()

        # 输入投影层：每种节点类型各一个，与 HGT 相同
        self.lin_dict = nn.ModuleDict()
        for node_type in train_data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        # HANConv 层
        # HANConv 内部自动处理所有元路径的节点级 + 语义级注意力
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HANConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    metadata=metadata,
                    heads=num_heads,
                    dropout=dropout,
                )
            )

        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

        # 用于累积每个 batch 的语义注意力权重
        # 结构: {layer_idx: List[Tensor]}，每个 Tensor shape = [num_metapaths]
        self._semantic_attn_accum = {i: [] for i in range(num_layers)}

    def forward(self, x_dict, edge_index_dict,return_weights=False):
        # 输入投影
        x_dict = {
            node_type: self.dropout(F.relu(self.lin_dict[node_type](x)))
            for node_type, x in x_dict.items()
            if node_type in self.lin_dict
        }

        for i, conv in enumerate(self.convs):
            if return_weights:
                out, attn_dict = conv(
                    x_dict, edge_index_dict,
                    return_semantic_attention_weights=True
                )
                # 优先取 drug，没有再取 disease
                if node_type1 in attn_dict and attn_dict[node_type1] is not None:
                    attn = attn_dict[node_type1]
                elif node_type2 in attn_dict and attn_dict[node_type2] is not None:
                    attn = attn_dict[node_type2]
                else:
                    attn = None
                if attn is not None:
                    self._semantic_attn_accum[i].append(attn.detach().cpu())
                x_dict = out
            else:
                x_dict = conv(x_dict, edge_index_dict)
        
            x_dict = {k: F.relu(v) for k, v in x_dict.items() if v is not None}

        
        drug_out    = self.lin(x_dict[node_type1])
        disease_out = self.lin(x_dict[node_type2])
        return F.relu(drug_out), F.relu(disease_out)
    
    def get_metapath_weights(self):
        """返回各层的平均语义注意力权重，shape: [num_layers, num_metapaths]"""
        result = {}
        for layer_idx, accum in self._semantic_attn_accum.items():
            if accum:
                result[f'layer_{layer_idx}'] = torch.stack(accum).mean(0)
        return result

    def reset_weight_accum(self):
        """每次评估前重置累积器"""
        for k in self._semantic_attn_accum:
            self._semantic_attn_accum[k] = []

# %%
class MLPPredictor(nn.Module):
    def __init__(self, channel_num, dropout):
        super().__init__()
        self.L1 = nn.Linear(channel_num * 3, channel_num)
        self.L2 = nn.Linear(channel_num, 1)
        self.bn = nn.BatchNorm1d(num_features=channel_num)
        self.dropout = nn.Dropout(0.2)

    # def forward(self, drug_embeddings, disease_embeddings):
    #     x = torch.cat((drug_embeddings, disease_embeddings), dim=1)
    #     x = F.relu(self.bn(self.L1(x)))
    #     x = self.dropout(x)
    #     x = self.L2(x)
    #     return x
    def forward(self, drug_emb, disease_emb):
        interaction = drug_emb * disease_emb
        x = torch.cat([drug_emb, disease_emb, interaction], dim=1)  # [B, 192]
        x = F.relu(self.bn(self.L1(x)))
        x = self.dropout(x)
        return self.L2(x)

# %%

def compute_loss(scores, labels):
    weight = torch.where(labels == 1,
                         (labels == 0).sum() / labels.shape[0],
                         (labels == 1).sum() / labels.shape[0])
    return F.binary_cross_entropy_with_logits(scores.squeeze(), labels, weight=weight)

# def compute_loss(scores, labels):
#     pos_weights = torch.clone(labels)
#     pos_weights[pos_weights == 1] = ((labels==0).sum() / labels.shape[0])
#     pos_weights[pos_weights == 0] = ((labels==1).sum() / labels.shape[0])
    
#     return F.binary_cross_entropy_with_logits(scores, labels, pos_weight=pos_weights)
# #     return F.binary_cross_entropy_with_logits(scores, labels)

# %%
def define_model(dropout):
    metadata = train_data.metadata()

    GNN = HAN(
        hidden_channels=64,
        out_channels=64,
        num_heads=8,
        num_layers=3,
        dropout=dropout,
        metadata=metadata,
    )

    pred = MLPPredictor(64, dropout)
    model = nn.Sequential(GNN, pred)
    model.to(device)

    return GNN, pred, model
# %%
def define_loaders(config):
    kwargs = {
        'batch_size': config['batch_size'],
        'num_workers': 2,               # Kaggle 2核，超过反而有进程竞争开销
        'persistent_workers': True,
        'pin_memory': True,             # 锁页内存，CPU→GPU 传输更快
    }
    
    train_loader = HGTLoader(train_data, num_samples=[config['num_samples']] * 3, shuffle=True, input_nodes=(node_type1, None), **kwargs)
    val_loader = HGTLoader(val_data, num_samples=[config['num_samples']] * 3, shuffle=True, input_nodes=(node_type1, None), **kwargs)

    return train_loader, val_loader

# %%
def edge_exists(edges, edge):
    edges = edges.to(device)
    edge = edge.to(device)
    return (edges == edge).all(dim=0).sum() > 0

# %% [markdown]
# ### Make batches.

# %%
def make_batch(batch):

    batch_size = batch[node_type1].batch_size
    edge_index = batch[(node_type1, rel, node_type2)]['edge_index']
    mask = batch[(node_type1, rel, node_type2)]['mask']

    batch_index = (edge_index[0] < batch_size)
    edge_index = edge_index[:, batch_index]
    mask = mask[batch_index]
    edge_label_index = edge_index[:, mask]
    pos_num = edge_label_index.shape[1]

    # ── 向量化负采样：一次性多采再过滤，避免 Python while 循环 ──
    pos_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    num_disease = batch[node_type2].x.shape[0]
    oversample = int(pos_num * 2) + 64          # 多采一些保证够用
    src_cand = torch.randint(0, batch_size,  (oversample,))
    dst_cand = torch.randint(0, num_disease, (oversample,))
    valid = [(s.item(), d.item()) for s, d in zip(src_cand, dst_cand)
             if (s.item(), d.item()) not in pos_set][:pos_num]
    # 极少情况下候选不足，循环补齐
    while len(valid) < pos_num:
        s = random.randint(0, batch_size - 1)
        d = random.randint(0, num_disease - 1)
        if (s, d) not in pos_set:
            valid.append((s, d))
    neg_edges = torch.tensor(valid, dtype=torch.long).t()   # [2, pos_num]
    # ─────────────────────────────────────────────────────────

    edge_label_index = torch.cat((edge_label_index, neg_edges), dim=1)
    edge_label = torch.cat((torch.ones(pos_num), torch.zeros(pos_num)))
    edge_index = edge_index[:, ~mask]

    batch[(node_type1, rel, node_type2)]['edge_index'] = edge_index
    batch[(node_type1, rel, node_type2)]['edge_label_index'] = edge_label_index
    batch[(node_type1, rel, node_type2)]['edge_label'] = edge_label

    batch[(node_type2, rel, node_type1)]['edge_index'] = torch.stack([
        edge_index[1].clone(),
        edge_index[0].clone()
    ])

    return batch

# %%

global_pos_set = set(zip(
    data[(node_type1, rel, node_type2)]['edge_index'][0].tolist(),
    data[(node_type1, rel, node_type2)]['edge_index'][1].tolist()
))  

def make_test_batch(batch):

    batch_size = batch[node_type1].batch_size
    edge_label_index = batch[(node_type1, rel, node_type2)]['edge_label_index']
    edge_label      = batch[(node_type1, rel, node_type2)]['edge_label']

    source, dest = [], []
    for i in range(edge_label_index.shape[1]):
        if (edge_label_index[0, i] in batch[node_type1]['id']
                and edge_label_index[1, i] in batch[node_type2]['id']
                and ((batch[node_type1]['id'] == edge_label_index[0, i])
                     .nonzero(as_tuple=True)[0]) < batch_size):
            if edge_label[i] == 1:
                source.append((batch[node_type1]['id'] == edge_label_index[0, i]).nonzero(as_tuple=True)[0])
                dest.append(  (batch[node_type2]['id'] == edge_label_index[1, i]).nonzero(as_tuple=True)[0])

    edge_label_index = torch.zeros(2, len(source), dtype=torch.long)
    edge_label_index[0] = torch.tensor(source)
    edge_label_index[1] = torch.tensor(dest)
    pos_num = edge_label_index.shape[1]

    # ── 向量化负采样 ─────────────────────────────────────────
    num_disease = batch[node_type2].x.shape[0]
    oversample  = int(pos_num * 2) + 64
    src_cand = torch.randint(0, batch_size,  (oversample,))
    dst_cand = torch.randint(0, num_disease, (oversample,))
    valid = []
    for s, d in zip(src_cand.tolist(), dst_cand.tolist()):
        orig_src = batch[node_type1]['id'][s].item()
        orig_dst = batch[node_type2]['id'][d].item()
        if (orig_src, orig_dst) not in global_pos_set:
            valid.append((s, d))
        if len(valid) == pos_num:
            break
    while len(valid) < pos_num:
        s = random.randint(0, batch_size - 1)
        d = random.randint(0, num_disease - 1)
        orig_src = batch[node_type1]['id'][s].item()
        orig_dst = batch[node_type2]['id'][d].item()
        if (orig_src, orig_dst) not in global_pos_set:
            valid.append((s, d))
    neg_edges = torch.tensor(valid, dtype=torch.long).t()
    # ─────────────────────────────────────────────────────────

    edge_label_index = torch.cat((edge_label_index, neg_edges), dim=1)
    edge_label = torch.cat((torch.ones(pos_num), torch.zeros(pos_num)))

    batch[(node_type1, rel, node_type2)]['edge_label_index'] = edge_label_index
    batch[(node_type1, rel, node_type2)]['edge_label'] = edge_label

    return batch

# %% [markdown]
# ### Train

# %%
def train(GNN, pred, model, loader, optimizer):
    model.train()
    total_examples = total_loss = 0
    for i, batch in enumerate(iter(loader)):
        optimizer.zero_grad()
        batch = make_batch(batch)
        batch = batch.to(device, non_blocking=True)
        edge_label_index = batch[(node_type1, rel, node_type2)]['edge_label_index']
        edge_label = batch[(node_type1, rel, node_type2)]['edge_label'].to(device, non_blocking=True)
        if edge_label.shape[0] == 0:
            continue
        
        drug_embeddings, disease_embeddings = GNN(batch.x_dict, batch.edge_index_dict)
        
        c = drug_embeddings[edge_label_index[0]]
        d = disease_embeddings[edge_label_index[1]]
        out = pred(c, d)[:, 0]
        loss = compute_loss(out, edge_label)
        loss.backward()
        optimizer.step()

        total_examples += edge_label_index.shape[1]
        # total_loss += float(loss) * edge_label_index.shape[1]
        # 方案 B (更常用): 直接使用 .item()，它会自动 detach
        total_loss += loss.item() * edge_label_index.shape[1]

    return total_loss / total_examples

# %% [markdown]
# ### Test

# %%
@torch.no_grad()
def test(GNN, pred, model, loader):
    model.eval()

    total_examples = total_correct = 0
    out, labels = torch.tensor([]).to(device), torch.tensor([]).to(device)
    source, dest = torch.tensor([]).to(device), torch.tensor([]).to(device)
    for batch in iter(loader):
        batch = make_test_batch(batch)
        batch = batch.to(device, non_blocking=True)
        drug_embeddings, disease_embeddings = GNN(batch.x_dict, batch.edge_index_dict)
        
        edge_label_index = batch[(node_type1, rel, node_type2)]['edge_label_index']
        edge_label = batch[(node_type1, rel, node_type2)]['edge_label'].to(device, non_blocking=True)
        
        if edge_label.shape[0] == 0:
            continue
                
        c = drug_embeddings[edge_label_index[0]]
        d = disease_embeddings[edge_label_index[1]]
        batch_out = pred(c, d)[:, 0]
        labels = torch.cat((labels, edge_label))
        out = torch.cat((out, batch_out))
        
        drugs = batch[node_type1]['id'][edge_label_index[0]]
        diseases = batch[node_type2]['id'][edge_label_index[1]]
        source = torch.cat((source, drugs))
        dest = torch.cat((dest, diseases))

    loss = compute_loss(out, labels)    
    return out, labels, source, dest, loss.cpu().numpy()


# %% [markdown]
# ### Run

# %%

# ─── 元路径重要性提取与可视化 ─────────────────────────────────────
METAPATH_LABELS = [
    "Drug→Dis→Dis→Drug",
    "Drug→Dis→Drug",
    "Drug→Pro→Dis→Drug",
    "Drug→Pro→Drug",
    "Dis→Dis→Dis",
    "Drug→Phe→Drug",
    "Dis→Phe→Dis",
    "Drug→Pro→Phe→Dis→Drug",
]

@torch.no_grad()
def extract_metapath_importance(GNN, loader, output_dir, epoch_label='best'):
    """
    在 eval 模式下跑一遍 loader，收集语义注意力权重，
    输出数值表格（CSV）+ 折线图（PNG）。
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    GNN.eval()
    GNN.reset_weight_accum()

    for batch in iter(loader):
        batch = make_test_batch(batch)
        batch = batch.to(device, non_blocking=True)
        # 开启权重收集
        GNN(batch.x_dict, batch.edge_index_dict, return_weights=True)

    weights_dict = GNN.get_metapath_weights()  # {layer_0: Tensor, layer_1: ...}

    if not weights_dict:
        print("[Warning] 未能提取到语义注意力权重，请确认 PyG 版本 >= 2.3")
        return

    num_layers = len(weights_dict)
    # ── 整理成 DataFrame ──────────────────────────────────────────
    rows = {}
    for layer_name, w in weights_dict.items():
        w_np = w.numpy()
        n = min(len(w_np), len(METAPATH_LABELS))
        rows[layer_name] = {METAPATH_LABELS[j]: round(float(w_np[j]), 6)
                            for j in range(n)}

    import pandas as pd
    df = pd.DataFrame(rows).T          # 行=layer，列=metapath
    df.index.name = 'layer'
    csv_path = os.path.join(output_dir, f'metapath_weights_{epoch_label}.csv')
    df.to_csv(csv_path)
    print(f"\n[元路径重要性] 已保存至 {csv_path}")
    print(df.to_string())

    # ── 分层柱状图 ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, num_layers,
                             figsize=(5 * num_layers, 4),
                             sharey=False)
    if num_layers == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors
    for ax, (layer_name, w) in zip(axes, weights_dict.items()):
        w_np = w.numpy()
        n = min(len(w_np), len(METAPATH_LABELS))
        labels = METAPATH_LABELS[:n]
        bars = ax.barh(labels, w_np[:n],
                       color=colors[:n], edgecolor='white', height=0.6)
        ax.set_xlabel('Semantic attention weight')
        ax.set_title(f'{layer_name}', fontsize=11)
        ax.set_xlim(0, max(w_np[:n]) * 1.25)
        for bar, val in zip(bars, w_np[:n]):
            ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=8)

    fig.suptitle(f'Meta-path Semantic Attention Weights ({epoch_label})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    bar_path = os.path.join(output_dir, f'metapath_weights_bar_{epoch_label}.png')
    fig.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[元路径重要性] 柱状图已保存至 {bar_path}")

    # ── 跨层折线图（若层数 > 1）────────────────────────────────────
    if num_layers > 1:
        fig2, ax2 = plt.subplots(figsize=(9, 4))
        layer_names = list(weights_dict.keys())
        first_w = next(iter(weights_dict.values())).numpy()
        n = min(len(first_w), len(METAPATH_LABELS))
        for j in range(n):
            vals = [weights_dict[ln].numpy()[j] for ln in layer_names]
            ax2.plot(layer_names, vals, marker='o',
                     label=METAPATH_LABELS[j], color=colors[j])
        ax2.set_ylabel('Semantic attention weight')
        ax2.set_title('Meta-path importance across layers', fontsize=12)
        ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        line_path = os.path.join(output_dir,
                                 f'metapath_weights_line_{epoch_label}.png')
        fig2.savefig(line_path, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f"[元路径重要性] 折线图已保存至 {line_path}")

    return df

def run(config):
    losses, val_losses = [], []
    best_val_loss = float('inf')
    best_epoch = 0

    # ── 早停参数 ──────────────────────────────────────────────
    patience         = config.get("early_stopping_patience", 20)
    min_delta        = config.get("early_stopping_min_delta", 1e-4)
    smooth_window    = config.get("early_stopping_smooth_window", 3)  # 平滑窗口
    no_improve_count = 0
    # ─────────────────────────────────────────────────────────

    train_loader, val_loader = define_loaders(config)
    GNN, pred, model = define_model(config['dropout'])
    
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                           T_max=config['epochs'], 
                                                           eta_min=0, 
                                                           last_epoch=-1)
    
    script_name = config.get("file_name")
    output_dir = os.path.join('/kaggle/working', script_name)
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in tqdm(range(config['epochs']), desc="Training Progress"):
        loss = train(GNN, pred, model, train_loader, optimizer)
        out, labels, source, dest, val_loss = test(GNN, pred, model, val_loader)
        
        write_to_out(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, ValLoss: {val_loss:.4f} \n', output_dir)
        losses.append(loss)
        val_losses.append(val_loss)
        plot_losses(losses, val_losses, output_dir)
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}: Learning Rate = {current_lr}")

        # ── 平滑验证损失（滑动窗口均值），消除随机负采样噪声 ──────
        smoothed_val_loss = float(
            sum(val_losses[-smooth_window:]) / len(val_losses[-smooth_window:])
        )
        print(f"  ValLoss={val_loss:.4f}  SmoothedValLoss={smoothed_val_loss:.4f}")

        # ── 保存 best model & 早停判断（基于平滑损失）─────────────
        if smoothed_val_loss < best_val_loss - min_delta:
            best_val_loss = smoothed_val_loss
            best_epoch = epoch
            no_improve_count = 0
            torch.save(model.state_dict(), os.path.join(output_dir, 'saved_model.pt'))
            print(f"  → Best model saved at epoch {epoch}, "
                  f"smoothed_val_loss={smoothed_val_loss:.4f}")
        else:
            no_improve_count += 1
            print(f"  → No improvement for {no_improve_count}/{patience} epochs "
                  f"(best smoothed_val_loss={best_val_loss:.4f} @ epoch {best_epoch})")
            if no_improve_count >= patience:
                write_to_out(
                    f'Early stopping triggered at epoch {epoch}. '
                    f'Best smoothed_val_loss={best_val_loss:.4f} at epoch {best_epoch}.\n',
                    output_dir
                )
                print(f"\n[早停] 在 epoch {epoch} 触发早停。"
                      f"最佳平滑验证损失 {best_val_loss:.4f} 出现在 epoch {best_epoch}。")
                break
        # ─────────────────────────────────────────────────────────

    # 训练结束后加载 best model 再做最终评估
    model.load_state_dict(torch.load(os.path.join(output_dir, 'saved_model.pt'), map_location=device))
    out, labels, source, dest, val_loss = test(GNN, pred, model, val_loader)
    AUPR(out, labels, output_dir)
    AUROC(out, labels, output_dir)
    # ── 元路径重要性输出
    extract_metapath_importance(GNN, val_loader, output_dir, epoch_label='best')

# %%
# run(config)





# %%
# 独立脚本：加载已保存的模型，提取并可视化元路径重要性
# 依赖：已在全局作用域中完成数据加载（data, train_data, val_data 等变量存在）

import os
import torch

# ─── 配置 ────────────────────────────────────────────────────────────────────
MODEL_PATH = '/kaggle/input/datasets/jkiobio/handr-2-data/HANDR-2/saved_model.pt'
OUTPUT_DIR = '/kaggle/working/HANDR-2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── 重建模型结构（与训练时完全一致）────────────────────────────────────────
dropout = config['dropout']   # 0.5，与训练 config 保持一致
GNN, pred, model = define_model(dropout)

# ─── 加载权重 ────────────────────────────────────────────────────────────────
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print(f" 模型已从 {MODEL_PATH} 加载完毕")

# ─── 重建 val_loader ─────────────────────────────────────────────────────────
_, val_loader = define_loaders(config)

# ─── 提取元路径重要性 ─────────────────────────────────────────────────────────
df_weights = extract_metapath_importance(
    GNN,
    val_loader,
    output_dir=OUTPUT_DIR,
    epoch_label='best'
)