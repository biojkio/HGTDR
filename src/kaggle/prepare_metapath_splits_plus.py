# %%

# 向量化改造版本
# prepare_metapath_splits_plus.py
#
# 功能：
#   1. 加载完整图 data（与训练代码相同的构建流程）
#   2. 在完整图上全局应用 AddMetaPaths
#   3. 加载原始5折 train1~5.pkl / val1~5.pkl
#   4. 将元路径边注入每折的 train/val 数据（保持原有结构不变）
#   5. 补充 mask（与训练代码逻辑完全一致：随机80%为True）
#   6. 保存为 train_mp1~5.pkl / val_mp1~5.pkl → ../data/CV_mp_data/
#
# 运行方式：python prepare_metapath_splits.py
# （在 Kaggle 上：把此文件作为单独 notebook cell 运行即可）



# %%
from torch_geometric.nn import HANConv, Linear
from torch_geometric.loader import HGTLoader
from torch_geometric.data import HeteroData
from torch_geometric.transforms import AddMetaPaths
import torch.nn.functional as F
import pickle
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import torch
import copy
import os
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# 0. 路径配置
# ─────────────────────────────────────────────────────────────────────────────
DATA_ROOT   = '/kaggle/input/datasets/jkiobio/hgtdr-data/data'
CV_SRC_DIR  = os.path.join(DATA_ROOT, 'CV data')          # 原始5折 pkl 所在目录
CV_DST_DIR  = '/kaggle/working/CV_mp_data'                         # 输出目录
os.makedirs(CV_DST_DIR, exist_ok=True)

N_FOLDS     = 5
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

node_type1 = 'drug'
node_type2 = 'disease'
rel        = 'indication'

# ─────────────────────────────────────────────────────────────────────────────
# 1. 元路径定义（与训练代码完全一致）
# ─────────────────────────────────────────────────────────────────────────────
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
    [('drug', 'drug_protein', 'gene/protein'), ('gene/protein', 'phenotype_protein', 'effect/phenotype'),
     ('effect/phenotype', 'disease_phenotype_positive', 'disease'), ('disease', 'indication', 'drug')],
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. 构建完整图 data（与训练代码完全相同的流程）
# ─────────────────────────────────────────────────────────────────────────────
print('=' * 60)
print('Step 1/4  构建完整图')
print('=' * 60)

primekg_file = os.path.join(DATA_ROOT, 'kg.csv')
df = pd.read_csv(primekg_file, sep=',')

valid_rows = (
    ((df['x_type'] == 'drug') & (df['y_type'] == 'disease')) |
    ((df['x_type'] == 'disease') & (df['y_type'] == 'drug'))
)
drug_disease_pairs = df[(df['relation'] == 'indication') & valid_rows]

x_mask = drug_disease_pairs['x_type'].isin([node_type1, node_type2])
y_mask = drug_disease_pairs['y_type'].isin([node_type1, node_type2])
all_entities = pd.concat([
    drug_disease_pairs.loc[x_mask, ['x_type', 'x_index']].rename(columns={'x_type': 'type', 'x_index': 'index'}),
    drug_disease_pairs.loc[y_mask, ['y_type', 'y_index']].rename(columns={'y_type': 'type', 'y_index': 'index'})
])
drugs    = all_entities[all_entities['type'] == node_type1]['index'].unique().tolist()
diseases = all_entities[all_entities['type'] == node_type2]['index'].unique().tolist()
valid_drugs    = set(drugs)
valid_diseases = set(diseases)

check_x = (
    (~df['x_type'].isin(['drug', 'disease'])) |
    ((df['x_type'] == 'drug')    & df['x_index'].isin(valid_drugs))    |
    ((df['x_type'] == 'disease') & df['x_index'].isin(valid_diseases))
)
check_y = (
    (~df['y_type'].isin(['drug', 'disease'])) |
    ((df['y_type'] == 'drug')    & df['y_index'].isin(valid_drugs))    |
    ((df['y_type'] == 'disease') & df['y_index'].isin(valid_diseases))
)
df_cleaned = df[check_x & check_y].reset_index(drop=True)

head_nodes = df_cleaned['x_type'] + '::' + df_cleaned['x_index'].astype(str)
tail_nodes = df_cleaned['y_type'] + '::' + df_cleaned['y_index'].astype(str)
new_df = pd.DataFrame({0: head_nodes, 1: df_cleaned['relation'], 2: tail_nodes})
df_trips = new_df.drop_duplicates()
triplets  = df_trips.values.tolist()
print(f'  三元组数量: {len(triplets)}')

# entity_dictionary
entity_dictionary = {}
for src, _, dest in triplets:
    for node in [src, dest]:
        n_type, n_id = node.split('::', 1)
        type_dict = entity_dictionary.setdefault(n_type, {})
        if node not in type_dict:
            type_dict[node] = len(type_dict)

# edge_dictionary
from collections import defaultdict
edge_dictionary = defaultdict(list)
for src, relation, dest in triplets:
    src_type  = src.split('::', 1)[0]
    dest_type = dest.split('::', 1)[0]
    src_int_id  = entity_dictionary[src_type][src]
    dest_int_id = entity_dictionary[dest_type][dest]
    edge_dictionary[(src_type, relation, dest_type)].append((src_int_id, dest_int_id))
edge_dictionary = dict(edge_dictionary)

# 节点特征
CHEMBERTA_DIM  = 767
PUBMEDBERT_DIM = 768
pubmedbert_df = pd.read_pickle(os.path.join(DATA_ROOT, 'pubmedbert_embeddings.pkl'))
smiles_df     = pd.read_pickle(os.path.join(DATA_ROOT, 'smiles_embeddings.pkl'))
pubmedbert_dict = dict(zip(pubmedbert_df['id'], pubmedbert_df['embedding']))
smiles_dict     = dict(zip(smiles_df['id'],     smiles_df['embedding']))

data = HeteroData()
for key in entity_dictionary.keys():
    num_nodes = len(entity_dictionary[key])
    dim = CHEMBERTA_DIM if key == 'drug' else PUBMEDBERT_DIM
    data[key].x  = torch.zeros((num_nodes, dim))
    data[key].id = torch.arange(num_nodes)

for key in edge_dictionary:
    data[key].edge_index = torch.transpose(
        torch.IntTensor(edge_dictionary[key]), 0, 1
    ).long().contiguous()

for node_type, mapping in entity_dictionary.items():

    if node_type == 'drug':
        source_dict = smiles_dict
    else:
        source_dict = pubmedbert_dict

    hgt_ids = []
    embeddings = []

    for entity_id, hgt_id in mapping.items():
        emb = source_dict.get(entity_id)  #  用 get 避免两次查找
        if emb is not None:
            hgt_ids.append(hgt_id)
            embeddings.append(emb)

    if embeddings:
        emb_tensor = torch.from_numpy(np.asarray(embeddings, dtype=np.float32))
        data[node_type].x[hgt_ids] = emb_tensor

print(f'  节点类型: {data.node_types}')
print(f'  原始边类型数: {len(data.edge_types)}')

# ─────────────────────────────────────────────────────────────────────────────
# 3. 在完整图上全局应用 AddMetaPaths
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('Step 2/4  全局应用 AddMetaPaths')
print('=' * 60)

# 保留未加元路径的完整图，供每折严格剔除 val 边后再计算元路径
full_data_no_mp = copy.deepcopy(data)

data = AddMetaPaths(METAPATHS, drop_orig_edge_types=False, weighted=True)(data)

print(f'  应用后边类型数: {len(data.edge_types)}')
# 找出新增的元路径边类型（关系名以 "metapath_" 开头是 PyG 的默认命名规则）
mp_etypes = [et for et in data.edge_types if 'metapath' in et[1]]
print(f'  新增元路径边类型:')
for et in mp_etypes:
    print(f'    {et}  →  {data[et].edge_index.shape[1]} 条边')

# ─────────────────────────────────────────────────────────────────────────────
# 4. 辅助函数：把完整图的元路径边注入到一个 fold 的 HeteroData 中
# ─────────────────────────────────────────────────────────────────────────────

def inject_metapath_edges(fold_data: HeteroData, full_data: HeteroData) -> HeteroData:
    """
    将 full_data 中所有元路径边类型（metapath_*）复制到 fold_data。
    fold_data 的原有边（indication 等监督边）保持不变。
    返回注入后的新 HeteroData（深拷贝，互不影响）。
    """
    new_data = copy.deepcopy(fold_data)
    for et in mp_etypes:
        new_data[et].edge_index = full_data[et].edge_index.clone()
        # 如果有边权重（weighted=True 时生成）一并复制
        if hasattr(full_data[et], 'edge_weight'):
            new_data[et].edge_weight = full_data[et].edge_weight.clone()
    return new_data


def add_mask(fold_data: HeteroData) -> HeteroData:
    """
    与训练代码完全一致的 mask 逻辑：
    随机选 80% 的 indication 边索引设为 True（作为监督边）。
    """
    drug_disease_num = fold_data[(node_type1, rel, node_type2)]['edge_index'].shape[1]
    mask_indices = random.sample(range(drug_disease_num), int(drug_disease_num * 0.8))

    fold_data[(node_type1, rel, node_type2)]['mask'] = torch.zeros(drug_disease_num, dtype=torch.bool)
    fold_data[(node_type1, rel, node_type2)]['mask'][mask_indices] = True

    fold_data[(node_type2, rel, node_type1)]['mask'] = torch.zeros(drug_disease_num, dtype=torch.bool)
    fold_data[(node_type2, rel, node_type1)]['mask'][mask_indices] = True

    return fold_data


# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数：剔除本折 val 的 indication 边后构造 source graph
# ─────────────────────────────────────────────────────────────────────────────
def make_fold_source_data(val_edge_index: torch.Tensor) -> HeteroData:
    """
    从完整图中剔除本折 val 的 indication 边，
    再应用 AddMetaPaths，得到严格无泄露的元路径图。
    其他所有边（drug_protein / disease_disease 等）保持完整不变。
    """
    src = copy.deepcopy(full_data_no_mp)

    val_set = set(zip(val_edge_index[0].tolist(), val_edge_index[1].tolist()))

    # 正向 indication 边：剔除 val 边
    full_ei = src[(node_type1, rel, node_type2)].edge_index



    val_ei = torch.tensor(list(val_set), dtype=full_ei.dtype, device=full_ei.device).t()
    num_nodes = int(max(full_ei.max(), val_ei.max())) + 1
    full_hash = full_ei[0] * num_nodes + full_ei[1]
    val_hash  = val_ei[0] * num_nodes + val_ei[1]
    # 修复后：
    keep = ~torch.isin(full_hash, val_hash)
    src[(node_type1, rel, node_type2)].edge_index = full_ei[:, keep]  # ← 补上这行

    # 反向 indication 边同步剔除
    full_ei_rev = src[(node_type2, rel, node_type1)].edge_index
    val_set_rev = {(d, s) for s, d in val_set}
    keep_rev = torch.tensor(
        [(s.item(), d.item()) not in val_set_rev
         for s, d in zip(full_ei_rev[0], full_ei_rev[1])],
        dtype=torch.bool
    )
    src[(node_type2, rel, node_type1)].edge_index = full_ei_rev[:, keep_rev]

    # 在剔除后的图上计算元路径
    src = AddMetaPaths(METAPATHS, drop_orig_edge_types=False, weighted=True)(src)
    return src


# ─────────────────────────────────────────────────────────────────────────────
# 5. 逐折处理并保存（方案C：每折独立计算元路径）
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('Step 3/4  处理5折数据（严格模式：每折剔除 val indication 边后计算元路径）')
print('=' * 60)

for fold in range(1, N_FOLDS + 1):
    print(f'\n  ── Fold {fold} ──')

    with open(os.path.join(CV_SRC_DIR, f'train{fold}.pkl'), 'rb') as f:
        train_fold = pickle.load(f)
    with open(os.path.join(CV_SRC_DIR, f'val{fold}.pkl'), 'rb') as f:
        val_fold = pickle.load(f)

    val_ei = val_fold[(node_type1, rel, node_type2)].edge_index
    print(f'    本折 val indication 边数（将被剔除后再算元路径）: {val_ei.shape[1]}')

    # 每折独立构造严格的元路径图
    fold_source_mp = make_fold_source_data(val_ei)

    # 注入元路径边
    train_mp = inject_metapath_edges(train_fold, fold_source_mp)
    val_mp   = inject_metapath_edges(val_fold,   fold_source_mp)

    # 给 train_data 加 mask
    train_mp = add_mask(train_mp)

    print(f'    注入后 train 边类型数: {len(train_mp.edge_types)}')
    print(f'    注入后 val   边类型数: {len(val_mp.edge_types)}')

    train_dst = os.path.join(CV_DST_DIR, f'train_mp{fold}.pkl')
    val_dst   = os.path.join(CV_DST_DIR, f'val_mp{fold}.pkl')

    with open(train_dst, 'wb') as f:
        pickle.dump(train_mp, f)
    with open(val_dst, 'wb') as f:
        pickle.dump(val_mp, f)

    print(f'    已保存 → {train_dst}')
    print(f'    已保存 → {val_dst}')



# ─────────────────────────────────────────────────────────────────────────────
# 6. 完成汇总
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('Step 4/4  完成')
print('=' * 60)
saved = sorted(os.listdir(CV_DST_DIR))
print(f'  输出目录: {CV_DST_DIR}')
print(f'  共 {len(saved)} 个文件:')
for fname in saved:
    fpath = os.path.join(CV_DST_DIR, fname)
    size_mb = os.path.getsize(fpath) / 1024 / 1024
    print(f'    {fname}  ({size_mb:.1f} MB)')


print('  已将 train1.pkl / val1.pkl 替换为 train_mp1.pkl / val_mp1.pkl')
