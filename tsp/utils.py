import torch
from torch_geometric.data import Data

def gen_distance_matrix(tsp_coordinates):
    '''
    Args:
        tsp_coordinates: torch tensor [n_nodes, 2] for node coordinates
    Returns:
        distance_matrix: torch tensor [n_nodes, n_nodes] for EUC distances
    '''
    n_nodes = len(tsp_coordinates)
    distances = torch.norm(tsp_coordinates[:, None] - tsp_coordinates, dim=2, p=2)
    distances[torch.arange(n_nodes), torch.arange(n_nodes)] = 1e9 # note here
    return distances

def gen_time_matrix(tsp_coordinates, speed=2.0, noise_std=0.0):
    """
    Sinh ma trận travel_time từ tọa độ.
    travel_time[i,j] = distance(i,j)/speed + GaussianNoise(0, noise_std)
    
    Args:
        tsp_coordinates: torch tensor [n_nodes, 2]
        speed: vận tốc trung bình (float >0)
        noise_std: độ lệch chuẩn của nhiễu Gaussian (float >=0)
    Returns:
        time_matrix: torch tensor [n_nodes, n_nodes]
    """
    n_nodes = len(tsp_coordinates)
    # tính ma trận khoảng cách Euclid
    distances = gen_distance_matrix(tsp_coordinates)  # [n_nodes,n_nodes]
    # chia cho tốc độ
    time_matrix = distances / speed
    # thêm nhiễu Gaussian nếu cần
    if noise_std > 0:
        time_matrix = time_matrix + torch.randn_like(time_matrix) * noise_std
    # loại bỏ self-loop
    idx = torch.arange(n_nodes, device=tsp_coordinates.device)
    time_matrix[idx, idx] = 1e9
    return time_matrix

def gen_pyg_data(tsp_coordinates, k_sparse, speed=2.0, noise_std=0.0):
    """
    Trả về:
      - pyg_data
      - distance matrix
      - time matrix
    """
    n_nodes = len(tsp_coordinates)
    distances   = gen_distance_matrix(tsp_coordinates)
    time_matrix = gen_time_matrix(tsp_coordinates, speed, noise_std)
    topk_values, topk_indices = torch.topk(distances, k=k_sparse, dim=1, largest=False)
    edge_index = torch.stack([
        torch.repeat_interleave(torch.arange(n_nodes, device=topk_indices.device),
                                repeats=k_sparse),
        torch.flatten(topk_indices)
    ])
    edge_attr = topk_values.reshape(-1, 1)
    pyg_data = Data(x=tsp_coordinates, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_data, distances, time_matrix

def load_val_dataset(n_node, k_sparse, device):
    val_list = []
    val_tensor = torch.load(f'../data/tsp/valDataset-{n_node}.pt')
    for instance in val_tensor:
        instance = instance.to(device)
        data, distances, tm = gen_pyg_data(instance, k_sparse=k_sparse)
        val_list.append((data, distances, tm))
    return val_list

def load_test_dataset(n_node, k_sparse, device):
    val_list = []
    val_tensor = torch.load(f'../data/tsp/testDataset-{n_node}.pt')
    for instance in val_tensor:
        instance = instance.to(device)
        data, distances, tm = gen_pyg_data(instance, k_sparse=k_sparse)
        val_list.append((data, distances, tm))
    return val_list

if __name__ == "__main__":
    pass