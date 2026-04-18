
# HGTDR-14-3修改版，嵌入全随机

from torch_geometric.nn import HGTConv, Linear
from torch_geometric.loader import HGTLoader
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit
from collections import defaultdict
import torch.nn.functional as F
import pickle
import torch.nn as nn
import numpy as np
import pandas as pd
# from utils import *
import random
import torch
import copy
import os
import sys
import time
from tqdm import tqdm

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # 多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
node_type1 = 'drug'
node_type2 = 'disease'
rel = 'indication'

config = {
    "seed": 42,
    "eval_seed": 99,          # 专门控制验证集负采样，与训练种子隔离
    "num_samples": 512,
    "batch_size": 164,
    "dropout": 0.5,
    "epochs": 300,
    "file_name": "HGTDR-17",
    # 早停参数
    "early_stopping_patience": 20,      # 容忍验证损失不改善的最大 epoch 数
    "early_stopping_min_delta": 1e-4,   # 认为"有改善"所需的最小降幅
    "early_stopping_smooth_window": 3   # 平滑窗口大小（消除随机负采样噪声）
}

set_seed(config["seed"])

primekg_file = '/kaggle/input/datasets/jkiobio/hgtdr-data/data/kg.csv'
df = pd.read_csv(primekg_file, sep =",")


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


entity_dictionary = {}

for src, _, dest in triplets:
    for node in [src, dest]:
        n_type, n_id = node.split('::', 1)
        
        # setdefault: 如果 key 不存在，初始化为空字典，并返回该字典
        type_dict = entity_dictionary.setdefault(n_type, {})
        
        # 如果实体不在字典中，赋予新 ID (当前长度)
        if node not in type_dict:
            type_dict[node] = len(type_dict)


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


# 节点类型组成维度 
# DrugNode2Vec + ChemBERTa 128 + 767 = 895 
# Non-DrugNode2Vec + PubMedBERT 128 + 768 = 896

# 初始化 HeteroData 并填充嵌入

CHEMBERTA_DIM  = 767
PUBMEDBERT_DIM = 768

# 加载嵌入文件
# pubmedbert_df = pd.read_pickle('/kaggle/input/datasets/jkiobio/hgtdr-data/data/pubmedbert_embeddings.pkl')
# smiles_df     = pd.read_pickle('/kaggle/input/datasets/jkiobio/hgtdr-data/data/smiles_embeddings.pkl')

# 创建字典
# pubmedbert_dict = dict(zip(pubmedbert_df['id'], pubmedbert_df['embedding']))
# smiles_dict     = dict(zip(smiles_df['id'],     smiles_df['embedding']))

# 初始化节点特征
data = HeteroData()
for key in entity_dictionary.keys():
    num_nodes = len(entity_dictionary[key])
    dim = CHEMBERTA_DIM if key == 'drug' else PUBMEDBERT_DIM
    data[key].x  = torch.zeros((num_nodes, dim))
    # data[key].x  = torch.randn((num_nodes, dim))  # 随机嵌入
    data[key].id = torch.arange(num_nodes)

# 添加边
for key in edge_dictionary:
    data[key].edge_index = torch.transpose(
        torch.IntTensor(edge_dictionary[key]), 0, 1
    ).long().contiguous()

# # 填充嵌入
# for node_type, mapping in tqdm(entity_dictionary.items(), desc='填充节点嵌入'):
#     for entity_id, hgt_id in mapping.items():

#         if node_type == 'drug':
#             if entity_id in smiles_dict:
#                 data[node_type].x[hgt_id, :] = torch.tensor( # 修改索引
#                     np.array(smiles_dict[entity_id], dtype=np.float32)
#                 )
#         else:
#             if entity_id in pubmedbert_dict:
#                 data[node_type].x[hgt_id, :] = torch.tensor( # 修改索引
#                     np.array(pubmedbert_dict[entity_id], dtype=np.float32)
#                 )

print('节点维度:')
for k in data.node_types:
    print(f'  {k}: {data[k].x.shape}')



# ═══════════════════════════════════════════════════════════════
# 替代 pkl：从 data 重新构建 train_data / val_data
# RandomLinkSplit 保证：验证监督边从消息传递图中彻底移除
# ═══════════════════════════════════════════════════════════════
from torch_geometric.transforms import RandomLinkSplit

# ── 1. 8:2 切分 indication 边 ──────────────────────────────────
transform = RandomLinkSplit(
    num_val=0.2,
    num_test=0.0,
    is_undirected=True,                             # 正向+反向边同步处理
    edge_types=[(node_type1, rel, node_type2)],
    rev_edge_types=[(node_type2, rel, node_type1)],
    add_negative_train_samples=False,               # 负样本由 make_batch 自己生成
)
train_data, val_data, _ = transform(data)

# ── 2. 补充节点 id 属性（make_batch/make_test_batch 依赖此字段）──
for split in [train_data, val_data]:
    for ntype in data.node_types:
        split[ntype].id = torch.arange(split[ntype].num_nodes)

# ── 3. 为 make_batch 添加 mask ─────────────────────────────────
#    在训练边中再取 80% 作为监督信号，其余 20% 留作消息传递
drug_disease_num = train_data[(node_type1, rel, node_type2)]['edge_index'].shape[1]
mask_idx = random.sample(range(drug_disease_num), int(drug_disease_num * 0.8))

for direction in [(node_type1, rel, node_type2), (node_type2, rel, node_type1)]:
    m = torch.zeros(drug_disease_num, dtype=torch.bool)
    m[mask_idx] = True
    train_data[direction]['mask'] = m

# ── 4. 泄露验证 ────────────────────────────────────────────────
val_el = val_data[(node_type1, rel, node_type2)].edge_label_index
val_ei = val_data[(node_type1, rel, node_type2)].edge_index
label_set = set(zip(val_el[0].tolist(), val_el[1].tolist()))
index_set  = set(zip(val_ei[0].tolist(), val_ei[1].tolist()))
overlap    = label_set & index_set

print(f"[重构验证] 训练 indication 边 : {drug_disease_num}")
print(f"[重构验证] 验证 supervision 边: {len(label_set)}")
print(f"[重构验证] 与消息传递图重叠   : {len(overlap)} 条 → {'⚠️ 泄露' if overlap else '✅ 无泄露'}")

# ── 5. 更新 global_pos_set（用重构后的完整 indication 边）───────
global_pos_set = set(zip(
    data[(node_type1, rel, node_type2)].edge_index[0].tolist(),
    data[(node_type1, rel, node_type2)].edge_index[1].tolist()
))



# %%
class HGT(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, dropout):
        super().__init__()

        self.lin_dict = nn.ModuleDict()
        for node_type in train_data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels[0])
            
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv = HGTConv(hidden_channels[i], hidden_channels[i+1], train_data.metadata(),
                           num_heads[i])
            self.convs.append(conv)
        
        self.lin = Linear(sum(hidden_channels[1:]), out_channels)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.dropout(self.lin_dict[node_type](x).relu_())
            for node_type, x in x_dict.items()
        }
        out = {}
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)

            if out=={}:
                out = copy.copy(x_dict)
            else:
                out = {
                    node_type: torch.cat((out[node_type], x_dict[node_type]), dim=1)
                    for node_type, x in x_dict.items()
                }

        return F.relu(self.lin(out[node_type1])), F.relu(self.lin(out[node_type2]))


class MLPPredictor(nn.Module):
    def __init__(self, channel_num, dropout):
        super().__init__()
        self.L1 = nn.Linear(channel_num * 3, channel_num)
        self.L2 = nn.Linear(channel_num, 1)
        self.bn = nn.BatchNorm1d(num_features=channel_num)
        self.dropout = nn.Dropout(0.2)

    def forward(self, drug_emb, disease_emb):
        interaction = drug_emb * disease_emb
        x = torch.cat([drug_emb, disease_emb, interaction], dim=1)  # [B, 192]
        x = F.relu(self.bn(self.L1(x)))
        x = self.dropout(x)
        return self.L2(x)

def compute_loss(scores, labels):
    weight = torch.where(labels == 1,
                         (labels == 0).sum() / labels.shape[0],
                         (labels == 1).sum() / labels.shape[0])
    return F.binary_cross_entropy_with_logits(scores.squeeze(), labels, weight=weight)

def define_model(dropout):
    GNN = HGT(hidden_channels=[64, 64, 64, 64],
              out_channels=64,
              num_heads=[8, 8, 8],
              num_layers=3,
              dropout=dropout)

    pred = MLPPredictor(64, dropout)
    model = nn.Sequential(GNN, pred)
    model.to(device)

    return GNN, pred, model

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
# ── 2. make_test_batch：用确定性种子，采完后恢复随机状态 ────────
def make_test_batch(batch, eval_seed=99):

    batch_size = batch[node_type1].batch_size
    edge_label_index = batch[(node_type1, rel, node_type2)]['edge_label_index']
    edge_label       = batch[(node_type1, rel, node_type2)]['edge_label']

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

    # ── 用正样本全局 ID 派生确定性 batch_seed ─────────────────
    # 同一批正样本无论第几个 epoch 调用，都能生成完全相同的负样本
    orig_srcs = batch[node_type1]['id'][edge_label_index[0]].tolist()
    orig_dsts = batch[node_type2]['id'][edge_label_index[1]].tolist()
    batch_seed = eval_seed
    for s, d in zip(sorted(orig_srcs), sorted(orig_dsts)):
        batch_seed = hash((batch_seed, s, d)) & 0xFFFFFFFF

    # ── 保存当前随机状态，切换到确定性种子 ──────────────────────
    py_rng_state    = random.getstate()
    torch_rng_state = torch.get_rng_state()
    cuda_rng_state  = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    random.seed(batch_seed)
    torch.manual_seed(batch_seed)

    # ── 负采样（逻辑与原 make_test_batch 完全一致）──────────────
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

    # ── 恢复随机状态（训练的随机性完全不受影响）────────────────
    random.setstate(py_rng_state)
    torch.set_rng_state(torch_rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)
    # ──────────────────────────────────────────────────────────

    neg_edges = torch.tensor(valid, dtype=torch.long).t()
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
# ── 3. test 函数：把 eval_seed 传进去 ────────────────────────
@torch.no_grad()
def test(GNN, pred, model, loader, eval_seed=99):
    model.eval()

    out, labels = torch.tensor([]).to(device), torch.tensor([]).to(device)
    source, dest = torch.tensor([]).to(device), torch.tensor([]).to(device)
    for batch in iter(loader):
        batch = make_test_batch(batch, eval_seed=eval_seed)   # ← 传入种子
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
        out    = torch.cat((out, batch_out))

        drugs    = batch[node_type1]['id'][edge_label_index[0]]
        diseases = batch[node_type2]['id'][edge_label_index[1]]
        source = torch.cat((source, drugs))
        dest   = torch.cat((dest,   diseases))

    loss = compute_loss(out, labels)
    return out, labels, source, dest, loss.cpu().numpy()
# %% [markdown]
# ### Run

# %%
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

     # ── 记录训练开始时间 ──────────────────────────────────────
    train_start_time = time.time()
    # ─────────────────────────────────────────────────────────
    
    for epoch in tqdm(range(config['epochs']), desc="Training Progress"):
        loss = train(GNN, pred, model, train_loader, optimizer)
        # ── 4. run 函数里的 test 调用加上 eval_seed ──────────────────
        # （两处，训练循环内和最终评估）
        out, labels, source, dest, val_loss = test(
            GNN, pred, model, val_loader,
            eval_seed=config['eval_seed']    # ← 加这个参数
        )
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


    # ── 记录训练结束时间并写入 out.txt ────────────────────────
    train_end_time = time.time()
    total_seconds = train_end_time - train_start_time
    h, rem = divmod(int(total_seconds), 3600)
    m, s = divmod(rem, 60)
    write_to_out(
        f'Total training time: {h:02d}h {m:02d}m {s:02d}s ({total_seconds:.1f}s), '
        f'finished at epoch {epoch}.\n',
        output_dir
    )
    # ─────────────────────────────────────────────────────────

    # 训练结束后加载 best model 再做最终评估
    model.load_state_dict(torch.load(os.path.join(output_dir, 'saved_model.pt'), map_location=device))
    
    # ── 4. run 函数里的 test 调用加上 eval_seed ──────────────────
    # （两处，训练循环内和最终评估）
    out, labels, source, dest, val_loss = test(
        GNN, pred, model, val_loader,
        eval_seed=config['eval_seed']    # ← 加这个参数
    )
    AUPR(out, labels, output_dir)
    AUROC(out, labels, output_dir)
    

# %%
# run(config)