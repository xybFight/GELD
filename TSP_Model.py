import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.mode = model_params['mode']
        self.encoder = TSP_Encoder(**model_params)
        self.decoder = TSP_Decoder(**model_params)
        self.encoded_nodes = None
        self.data = None
        self.dis_matrix = None
        self.region = None

    def pre_forward(self, state, Norm=True):
        data = state.data
        if Norm:
            min_val, _ = torch.min(data, dim=1, keepdim=True)
            max_val, _ = torch.max(data, dim=1, keepdim=True)
            max_diff_values, _ = torch.max(max_val - min_val, dim=-1)
            norm_data = (data - min_val) / max_diff_values.unsqueeze(2)
            self.data = norm_data
        else:
            # for ablation study
            self.data = data

        if data.size(1) > 10000:
            # find knn with low precision
            self.decoder.data = self.data
            self.dis_matrix = compute_distance_matrix(self.data, 10000)
        else:
            # high precision for reuse
            self.dis_matrix = torch.cdist(self.data, self.data, p=2)
            self.dis_matrix.diagonal(dim1=-2, dim2=-1).zero_()
        self.region = map_coordinates_to_regions(self.data)

    def forward(self, state, selected_node_list, solution, current_step, repair=False, beam_search=False, beam_size=16,
                lib=False):
        batch_size_V = selected_node_list.size(0)

        if self.mode == 'train' and self.training:
            encoded_nodes = self.encoder(self.data, self.region)
            probs = self.decoder(encoded_nodes, selected_node_list, self.dis_matrix)
            selected_student = probs.argmax(dim=1)
            selected_teacher = solution[:, current_step - 1]
            prob = probs[torch.arange(batch_size_V)[:, None], selected_teacher[:, None]].reshape(batch_size_V, 1)
            return selected_teacher, prob, 1, selected_student

        if self.mode == 'test' or not self.training:
            if repair == False:
                if current_step <= 1 and not lib:
                    self.encoded_nodes = self.encoder(self.data, self.region)

                if not beam_search:
                    probs = self.decoder(self.encoded_nodes, selected_node_list, self.dis_matrix)
                    selected_student = probs.argmax(dim=1)
                    selected_teacher = selected_student
                    prob = 1
                    return selected_teacher, prob, 1, selected_student
                else:
                    probs = self.decoder(self.encoded_nodes, selected_node_list, self.dis_matrix, beam_search=True,
                                         beam_size=beam_size)
                    return None, probs, 1, None
            else:
                if current_step <= 2:
                    self.encoded_nodes = self.encoder(self.data, self.region)
                probs = self.decoder(self.encoded_nodes, selected_node_list, self.dis_matrix)
                selected_student = probs.argmax(dim=1)
                selected_teacher = selected_student
                prob = 1
                return selected_teacher, prob, 1, selected_student


def compute_distance_matrix(coords, block_size=5000):
    B, N, D = coords.size()
    distance_matrix = torch.empty(B, N, N, dtype=torch.float16, device='cuda')
    for i in range(0, N, block_size):
        end_i = min(i + block_size, N)
        for j in range(0, N, block_size):
            end_j = min(j + block_size, N)
            block_coords_i = coords[:, i:end_i]
            block_coords_j = coords[:, j:end_j]
            block_distances = torch.cdist(block_coords_i, block_coords_j, p=2)
            block_distances.diagonal(dim1=-2, dim2=-1).zero_()
            distance_matrix[:, i:end_i, j:end_j] = block_distances.to(dtype=torch.float16)

    return distance_matrix


def map_coordinates_to_regions(coordinates, grid_size=3):
    region_indices = torch.floor(coordinates * grid_size).long()
    region_indices = torch.clamp(region_indices, min=0, max=grid_size - 1)
    flat_indices = region_indices[:, :, 0] * grid_size + region_indices[:, :, 1]
    return flat_indices


########################################
# ENCODER
########################################
class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        self.embedding = nn.Linear(2, embedding_dim, bias=True)
        self.layers_global = nn.ModuleList([EncoderLayer(**model_params) for _ in range(1)])

    def forward(self, data, region):
        embedded_input = self.embedding(data)
        out = embedded_input
        for index in range(1):
            out = self.layers_global[index](out, region)
        return out


class TSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        self.data = None
        self.embedding_first_node1 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node1 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.layers_global = nn.ModuleList([DecoderLayer(**model_params) for _ in range(6)])
        self.Linear_final = nn.Linear(embedding_dim, 1, bias=True)

    def _get_new_data(self, data, selected_node_list, prob_size, B_V, dis_matrix, beam_search=False, beam_size=16):
        list = selected_node_list
        if beam_search:
            batch_size = B_V * beam_size
        else:
            batch_size = B_V

        new_list = torch.arange(prob_size)[None, :].repeat(batch_size, 1)

        new_list_len = prob_size - list.shape[1]

        index_2 = list.type(torch.long)

        index_1 = torch.arange(batch_size, dtype=torch.long)[:, None].expand(batch_size, index_2.shape[1])

        new_list[index_1, index_2] = -2
        # local selection
        num_k = 99

        if self.training:
            unselect_list = new_list[torch.gt(new_list, -1)].view(B_V, new_list_len)
        else:
            if new_list_len <= num_k:
                unselect_list = new_list[torch.gt(new_list, -1)].view(batch_size, new_list_len)
            else:
                last_node = selected_node_list[:, -1]
                if beam_search:
                    last_node = last_node.view(B_V, beam_size, 1).expand(-1, -1, prob_size)
                    distance = dis_matrix.gather(1, last_node).view(B_V * beam_size, -1)
                else:
                    distance = dis_matrix.gather(1, last_node.view(batch_size, 1, 1).expand(-1, -1, prob_size)).squeeze(
                        1)
                mask = torch.zeros_like(distance)
                mask[index_1, selected_node_list] = 1e2
                unselect_list = torch.topk(distance + mask, dim=1, k=num_k, largest=False).indices

        # ----------------------------------------------------------------------------
        new_data = data
        emb_dim = data.shape[-1]
        if beam_search:
            new_data = new_data.unsqueeze(1).expand(-1, beam_size, -1, -1)
            data_global = torch.gather(new_data, 2,
                                       unselect_list.view(B_V, beam_size, -1, 1).expand(-1, -1, -1, emb_dim))
            data_global = data_global.view(batch_size, -1, emb_dim)
        else:
            data_global = torch.gather(new_data, 1, unselect_list.unsqueeze(2).expand(-1, -1, emb_dim))

        return data_global, unselect_list

    def _get_encoding(self, encoded_nodes, node_index_to_pick, beam_search=False, beam_size=16):

        batch_size = node_index_to_pick.size(0)
        pomo_size = node_index_to_pick.size(1)
        embedding_dim = encoded_nodes.size(2)
        if beam_search:
            encoded_nodes = encoded_nodes.unsqueeze(1).expand(batch_size // beam_size, beam_size, -1, embedding_dim)
            gathering_index = node_index_to_pick.view(batch_size // beam_size, beam_size, pomo_size, 1).expand(
                batch_size // beam_size, beam_size, pomo_size, embedding_dim)
            picked_nodes = encoded_nodes.gather(dim=2, index=gathering_index)
            picked_nodes = picked_nodes.view(batch_size, pomo_size, embedding_dim)
        else:
            gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
            picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)

        return picked_nodes

    def forward(self, data, selected_node_list, dis_matrix, beam_search=False, beam_size=16):
        batch_size_V = data.shape[0]  # B

        problem_size = data.shape[1]

        new_data = data

        global_info, unselect_list = self._get_new_data(new_data, selected_node_list, problem_size, batch_size_V,
                                                        dis_matrix, beam_search=beam_search, beam_size=beam_size)

        first_and_last_node = self._get_encoding(new_data, selected_node_list[:, [0, -1]], beam_search=beam_search,
                                                 beam_size=beam_size)
        embedded_first_node_ = first_and_last_node[:, 0]
        embedded_last_node_ = first_and_last_node[:, 1]
        # ------------------------------------------------
        # ------------------------------------------------
        embedded_first_node_1 = self.embedding_first_node1(embedded_first_node_)
        embedded_last_node_1 = self.embedding_last_node1(embedded_last_node_)
        out_global = torch.cat((embedded_first_node_1.unsqueeze(1), global_info, embedded_last_node_1.unsqueeze(1)),
                               dim=1)
        unselect_list = torch.cat(
            (selected_node_list[:, 0].unsqueeze(1), unselect_list, selected_node_list[:, -1].unsqueeze(1)),
            dim=-1).type(torch.long)

        if problem_size > 10000:
            # re-compute distance matrix with high precision
            if beam_search:
                index_un = unselect_list.view(batch_size_V, beam_size, -1, 1).expand(batch_size_V, beam_size, -1, 2)
                cor = self.data.view(batch_size_V, 1, -1, 2).expand(batch_size_V, beam_size, -1, 2).gather(dim=2,
                                                                                                           index=index_un).view(
                    batch_size_V * beam_size, -1, 2)
                dis_matrix_un = torch.cdist(cor, cor, p=2)
                dis_matrix_un.diagonal(dim1=-2, dim2=-1).zero_()

            else:
                index_un = unselect_list.unsqueeze(2).expand(batch_size_V, -1, 2)
                cor = self.data.gather(dim=1, index=index_un)
                dis_matrix_un = torch.cdist(cor, cor, p=2)
                dis_matrix_un.diagonal(dim1=-2, dim2=-1).zero_()
        else:
            # reuse high precision distance matrix
            if beam_search:
                index_1_ = torch.arange(batch_size_V, dtype=torch.long).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                index_2_ = torch.arange(beam_size, dtype=torch.long).unsqueeze(0).unsqueeze(2).unsqueeze(3)
                index_un = unselect_list.view(batch_size_V, beam_size, -1, 1)
                dis_matrix = dis_matrix.unsqueeze(1).expand(batch_size_V, beam_size, problem_size, problem_size)
                dis_matrix_un = dis_matrix[index_1_, index_2_, index_un, index_un.transpose(2, 3)].view(
                    batch_size_V * beam_size, unselect_list.shape[-1], unselect_list.shape[-1])
            else:
                index_1_ = torch.arange(batch_size_V, dtype=torch.long).unsqueeze(1).unsqueeze(2)
                index_un = unselect_list.unsqueeze(1)
                dis_matrix_un = dis_matrix[index_1_, index_un, index_un.transpose(1, 2)]

        for index in range(6):
            out_global = self.layers_global[index](out_global, dis_matrix_un)
        out = self.Linear_final(out_global).squeeze(-1)
        out[:, [0, -1]] = out[:, [0, -1]] + float('-inf')

        props = F.softmax(out, dim=-1)
        props = props[:, 1:-1]

        index_small = torch.le(props, 1e-5)
        props_clone = props.clone()
        props_clone[index_small] = props_clone[index_small] + torch.tensor(1e-7, dtype=props_clone[
            index_small].dtype)
        props = props_clone

        # attention mechanism from ICAM may lead to NAN
        if props.isnan().any():
            flag = torch.isnan(props)
            row_indice = flag.any(dim=1).nonzero(as_tuple=True)[0]
            props[flag] = 1e-5
            props[row_indice, 0] = 1

        batch_size_V = out.size(0)
        new_props = torch.zeros(batch_size_V, problem_size) + 1e-5
        index_1_ = torch.arange(batch_size_V, dtype=torch.long)[:, None]
        new_props[index_1_, unselect_list[:, 1:-1]] = props
        new_props[index_1_, selected_node_list] = 1e-20
        return new_props


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        self.attentionlayer = SparseAttention(model_params=model_params)
        self.multi_head_combine = nn.Linear(embedding_dim, embedding_dim)
        self.feedForward = Feed_Forward_Module(**model_params)

    def forward(self, input1, index_region):
        hidden_states = input1
        multi_head_out = self.multi_head_combine(self.attentionlayer(hidden_states, index_region))
        out1 = input1 + multi_head_out
        hidden_states = out1
        hidden_states = self.feedForward(hidden_states)
        out3 = out1 + hidden_states
        return out3


class DecoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        self.input_layernorm = RMSNorm(embedding_dim)
        self.post_attention_layernorm = RMSNorm(embedding_dim)

        self.attentionlayer = SparseAttention_AFM(model_params=model_params)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim, bias=False)

        self.feedForward = Feed_Forward_Module(**model_params)

    def forward(self, input1, dis_matrix):
        hidden_states = self.input_layernorm(input1)

        multi_head_out = self.multi_head_combine(self.attentionlayer(hidden_states, dis_matrix))
        out1 = input1 + multi_head_out
        hidden_states = self.post_attention_layernorm(out1)

        hidden_states = self.feedForward(hidden_states)
        out3 = out1 + hidden_states
        return out3


def reshape_by_heads(qkv, head_num):
    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)

    q_transposed = q_reshaped.transpose(1, 2)

    return q_transposed


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)b
        return self.W2(F.relu(self.W1(input1)))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # RMSNorm
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class SparseAttention_AFM(nn.Module):
    # From ICAM
    def __init__(self, model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, dis_matrix):
        head_num = self.model_params['head_num']
        q = reshape_by_heads(self.Wq(x), head_num=head_num)
        k = reshape_by_heads(self.Wk(x), head_num=head_num)
        v = reshape_by_heads(self.Wv(x), head_num=head_num)

        dis_weight = torch.exp(-self.alpha * np.log2(dis_matrix.size(1)) * dis_matrix)
        k_weight = torch.exp(k)
        weight_1 = torch.einsum('bij, bhik->bhjk', dis_weight, torch.mul(k_weight, v))
        weight_2 = torch.einsum('bij, bhik->bhjk', dis_weight, k_weight)
        weight_3 = torch.div(weight_1, weight_2)
        q_weight = torch.sigmoid(q)
        out = torch.mul(q_weight, weight_3)
        out_transposed = out.transpose(1, 2)
        out = out_transposed.reshape(x.shape)
        return out


class SparseAttention(nn.Module):
    # RALA
    def __init__(self, model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x, index_region):
        head_num = self.model_params['head_num']
        agent_matrix = self.Wq(x)
        q = reshape_by_heads(agent_matrix, head_num=head_num)
        k = reshape_by_heads(self.Wk(x), head_num=head_num)
        v = reshape_by_heads(self.Wv(x), head_num=head_num)
        b_, k_ = k.size(0), k.size(3)
        region_mask = torch.zeros(b_, 9, requires_grad=False, dtype=torch.float)
        region_mask.scatter_add_(dim=1, index=index_region, src=torch.ones(index_region.shape))

        # averaging
        region_mask = torch.where(region_mask == 0, torch.tensor(1, dtype=torch.float, device=region_mask.device),
                                  region_mask)
        region_sums = torch.zeros(b_, 9, x.size(2))
        region_sums.scatter_add_(dim=1, index=index_region.unsqueeze(-1).expand(agent_matrix.shape), src=agent_matrix)
        agent = reshape_by_heads(region_sums / region_mask.unsqueeze(-1), head_num=head_num)

        score = torch.matmul(q, agent.transpose(2, 3)) * (k_) ** (-0.5)
        attention1 = F.softmax(score, dim=-1)
        score_k = torch.matmul(agent, k.transpose(2, 3)) * (k_) ** (-0.5)
        attention2 = F.softmax(score_k, dim=-1)
        out = torch.matmul(attention2, v)
        out = torch.matmul(attention1, out)
        out_transposed = out.transpose(1, 2)
        out = out_transposed.reshape(x.shape)
        return out
