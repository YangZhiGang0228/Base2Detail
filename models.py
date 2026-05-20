from include import *
import torch.nn.functional as F
import torch
import numpy as np
from sparse_loss import sparse_s_loss_sum_count


class MLP(torch.nn.Module):
    def __init__(self, Din, Dhid, Dout):
        super(MLP, self).__init__()
        self.layerIn = torch.nn.Linear(Din, Dhid[0])
        self.hidden = torch.nn.ModuleList()
        for ii in range(len(Dhid)-1):
            self.hidden.append(torch.nn.Linear(Dhid[ii], Dhid[ii+1]))
        self.layerOut = torch.nn.Linear(Dhid[-1], Dout)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.layerIn(x)
        x = self.relu(x)
        for ii in range(len(self.hidden)):
            x = self.hidden[ii](x)
            x = self.relu(x)
        x = self.layerOut(x)
        return x

class SubdNet(torch.nn.Module):
    def __init__(self, params):
        super(SubdNet, self).__init__()
        self.Dout_feature = params['Dout']
        self.numSubd = params["numSubd"]
        self.D_lifted = params['Lifted']
        self.PE_dim = 16
        self.V_feat_dim = 9+self.D_lifted + 2*self.PE_dim
        self.V_view_dim = 9+self.D_lifted+self.PE_dim
        self.V_local_dim = self.D_lifted+self.Dout_feature
        self.sparse_a = float(params.get("sparse_a", 1e-2))
        
        self.net_init = MLP(9 + 6, params['h_initNet'], self.Dout_feature)
        self.net_lift = MLP(self.Dout_feature, params['h_liftNet'], self.D_lifted)
        self.net_A = MLP(self.V_feat_dim, params['h_matrixNet'], 1)
        self.net_L = MLP(self.V_feat_dim, params['h_matrixNet'], 1)
        self.net_view_weight = MLP(self.V_view_dim, params['h_matrixNet'], 1)
        self.net_feature = MLP(self.V_local_dim, params['h_FNet'], self.Dout_feature)
        
        self.pool = torch.nn.AvgPool2d((2,1)) 
        
        # 全局缓存拓扑索引，避免循环中重复创建
        self.topo_cache = {}

    def halfEdgePool(self, fhe):
        fhe = fhe.unsqueeze(0).unsqueeze(0)
        fe = self.pool(fhe)
        fe = fe.squeeze(0).squeeze(0)
        return fe

    def flapNormalization(self, hf_pos):
        V = hf_pos 
        F = torch.tensor([[0,1,2],[1,0,3]], device=hf_pos.device) 
        b1 = (V[:,1,:] - V[:,0,:]) / (torch.norm(V[:,1,:] - V[:,0,:], dim=1).unsqueeze(1) + 1e-8)
        vec1 = V[:,F[:,1],:] - V[:,F[:,0],:]
        vec2 = V[:,F[:,2],:] - V[:,F[:,0],:]
        FN = torch.cross(vec1, vec2, dim=-1) 
        FNnorm = torch.norm(FN, dim=2)
        FN = FN / (FNnorm.unsqueeze(2) + 1e-8)
        eN = FN[:,0,:] + FN[:,1,:]
        b3 = eN / (torch.norm(eN, dim=1).unsqueeze(1) + 1e-8)
        b2 = torch.cross(b3, b1, dim=-1)
        return torch.cat((b1.unsqueeze(1), b2.unsqueeze(1), b3.unsqueeze(1)), dim=1)

    def _get_topo_indices(self, n, device):
        key = (n, device)
        if key not in self.topo_cache:
            i_idx = torch.arange(n+1, device=device).view(n+1, 1).expand(n+1, n+1)
            j_idx = torch.arange(n+1, device=device).view(1, n+1).expand(n+1, n+1)
            
            cyclic_idx = torch.zeros(n+1, n+1, dtype=torch.long, device=device)
            cyclic_idx[1:, 1:] = 1 + (j_idx[1:, 1:] - 1 + i_idx[1:, 1:] - 1) % n
            cyclic_idx[:, 0] = torch.arange(n+1, device=device)
            cyclic_idx[0, :] = 0
            
            k_idx = torch.arange(n, device=device).view(n, 1)
            c_idx = torch.arange(n, device=device).view(1, n)
            j_gather = (c_idx - k_idx) % n
            
            self.topo_cache[key] = {
                'cyclic_idx': cyclic_idx,
                'triu_idx_A': torch.triu_indices(n+1, n+1, offset=1, device=device),
                'triu_idx_L': torch.triu_indices(n+1, n+1, offset=0, device=device),
                'diag_idx': torch.arange(1, n+1, device=device),
                'j_gather': j_gather
            }
        return self.topo_cache[key]

    def forward(self, fv, mIdx, HFs, poolMats, DOFs, Us, Lambdas, groupedLocalData):
        outputs = []
        consist_losses = []
        device = fv.device
        sparse_loss_sum = fv.new_tensor(0.0)
        sparse_loss_count = 0
        
        pos = fv[:, :3]
        
        # ==================== Step 1: 初始特征提取 (Subd0) ====================
        curr_feat = torch.zeros(pos.size(0), self.Dout_feature, device=device)
        grouped_data_0 = groupedLocalData[mIdx][0] 
        
        for deg, data in grouped_data_0.items():
            v0_idx, flap_idx = data['v0'], data['flap_idx']
            B, n = v0_idx.size(0), deg
            
            VN_global = fv[v0_idx, 3:6]
            LC_global = fv[v0_idx, 6:9]
            
            ref_hf_idx = flap_idx[:, :, 0, :] 
            local_frames = self.flapNormalization(pos[ref_hf_idx].view(B*n, 4, 3)) 
            
            VN_global_exp = VN_global.unsqueeze(1).expand(B, n, 3).reshape(B*n, 1, 3)
            LC_global_exp = LC_global.unsqueeze(1).expand(B, n, 3).reshape(B*n, 1, 3)
            VN_local = torch.bmm(VN_global_exp, local_frames.transpose(1, 2)).view(B, n, 3)
            LC_local = torch.bmm(LC_global_exp, local_frames.transpose(1, 2)).view(B, n, 3)
            v0_6d = torch.cat([VN_local, LC_local], dim=-1) # (B, n, 6)
            
            v123_global = pos[flap_idx[:, :, 0, 1:4]] # 仅取第一个, 形状 (B, n, 3, 3)
            v123_centered = v123_global - pos[v0_idx].view(B, 1, 1, 3)
            v123_local_flat = torch.bmm(v123_centered.view(B*n, 3, 3), local_frames.transpose(1, 2))
            v123_local = v123_local_flat.view(B, n, 9) # 展平为 (B, n, 9),n种坐标系
            
            local_init_input = torch.cat([v123_local, v0_6d], dim=-1) # (B, n, 15)，每个坐标系下只预测一次高维坐标
            feat_init_flat = self.net_init(local_init_input.view(-1, local_init_input.shape[-1]))#(B * n, self.Dout_feature)
            feat_init = feat_init_flat.view(B, n, self.Dout_feature)
            
            # 沿 n 个局部视角求平均
            curr_feat[v0_idx] = feat_init.mean(dim=1)

        outputs.append(pos)

        # ==================== Step 2: 细分循环 ====================
        for ii in range(self.numSubd):
            nV_even = DOFs[mIdx][ii].size(0)
            nHE = HFs[mIdx][ii].size(0)
            
            new_pos_even = torch.zeros(nV_even, 3, device=device)
            new_feat_even = torch.zeros(nV_even, self.Dout_feature, device=device)
            new_pos_odd_he = torch.zeros(nHE, 3, device=device)
            new_feat_odd_he = torch.zeros(nHE, self.Dout_feature, device=device)
            
            grouped_data = groupedLocalData[mIdx][ii] 
            
            for deg, data in grouped_data.items():
                v0_idx, flap_idx, he_idx = data['v0'], data['flap_idx'], data['he_idx']
                pos_enc = data['pos_enc'] 
                B, n = v0_idx.size(0), deg
                neighbor_idx = flap_idx[:, :, :, 1] 
                topo = self._get_topo_indices(n, device)
                
                ref_hf_pos = pos[flap_idx[:, :, 0, :]] 
                local_frames = self.flapNormalization(ref_hf_pos.view(B*n, 4, 3))  

                v1_centered = pos[neighbor_idx] - pos[v0_idx].view(B, 1, 1, 3)
                v1_local_flat = torch.bmm(v1_centered.view(B*n, n, 3), local_frames.transpose(1, 2))
                v1_local = v1_local_flat.view(B, n, n, 3)
                v123_global = pos[flap_idx[:, :, :, 1:4]] 
                v123_centered = v123_global - pos[v0_idx].view(B, 1, 1, 1, 3)
                v123_local_flat = torch.bmm(v123_centered.view(B*n, n*3, 3), local_frames.transpose(1, 2))
                v123_local = v123_local_flat.view(B, n, n, 9)
                ref_9d_context = v123_local[:, :, 0, :]

                feat_neighbor = curr_feat[neighbor_idx] #[B,n,n,Dout]
                feat_v0 = curr_feat[v0_idx] #[B,Dout]
                neighbor_idx_unique = flap_idx[:, 0, :, 1] #这里因为取maxpool且高维特征与坐标系无关所以只选取第0个坐标系
                feat_neighbor_unique = curr_feat[neighbor_idx_unique] # (B, n, Dout)
                combined_feat = torch.cat([feat_v0.unsqueeze(1), feat_neighbor_unique], dim=1) #（B,n+1,Dout)
                combined_feat_high = self.net_lift(combined_feat.view(-1, self.Dout_feature)) #(B*n+1,Dout)
                combined_feat_high = combined_feat_high.view(B, n + 1, self.D_lifted) # (B, n+1, D_lifted)
                ring1_high = combined_feat_high.max(dim=1)[0] #(B,D_lifted)
                updated_v0_feat = self.net_feature(torch.cat([ring1_high, feat_v0], dim=-1)) # (B, Dout)

                pe_v0 = torch.zeros(B, 1, self.PE_dim, device=device) 
                pos_enc_exp = pos_enc.unsqueeze(0).expand(B, n, self.PE_dim) # (B, n, PE_dim)
                pe_all = torch.cat([pe_v0, pos_enc_exp], dim=1) # (B, n+1, PE_dim)
                # 两两成对的顶点位置编码
                pe_row = pe_all.unsqueeze(2).expand(B, n+1, n+1, self.PE_dim)
                pe_col = pe_all.unsqueeze(1).expand(B, n+1, n+1, self.PE_dim)
                pe_pairs = torch.cat([pe_row, pe_col], dim=-1) # (B, n+1, n+1, 2*PE_dim)

                # ------ 预测矩阵 A ------
                # 输入：当前坐标系下首个half-flap(9D) + 1环高维特征(D_lifted)
                ctx_common = torch.cat([ref_9d_context, ring1_high.unsqueeze(1).expand(B, n, self.D_lifted)], dim=-1) # (B, n, 9+D_lifted)
                
                ctx_A_exp = ctx_common.view(B, n, 1, 1, -1).expand(B, n, n+1, n+1, 9 + self.D_lifted)
                pe_pairs_exp = pe_pairs.unsqueeze(1).expand(B, n, n+1, n+1, 2 * self.PE_dim)
                Pairs_A = torch.cat([ctx_A_exp, pe_pairs_exp], dim=-1) # (B, n, n+1, n+1, 9+D_lifted+2*PE_dim)

                A_raw_flat = self.net_A(Pairs_A.view(-1, Pairs_A.shape[-1])) #(B*n*n+1*n+1,1)
                A_raw = A_raw_flat.view(B, n, n+1, n+1).squeeze(-1)
                A = torch.zeros(B, n, n+1, n+1, device=device)
                A[:, :, topo['triu_idx_A'][0], topo['triu_idx_A'][1]] = A_raw[:, :, topo['triu_idx_A'][0], topo['triu_idx_A'][1]]
                

                ctx_L_exp = ctx_common.unsqueeze(2).expand(B, n, n, 9 + self.D_lifted)
                pe_j_exp = pos_enc.unsqueeze(1).expand(B, n, n, self.PE_dim) 
                Pairs_w = torch.cat([ctx_L_exp, pe_j_exp], dim=-1) # (B, n, n, 9+D_lifted+PE_dim)
                view_weight_flat = self.net_view_weight((Pairs_w.view(-1, Pairs_w.shape[-1])))
                perc_diag = torch.sigmoid(view_weight_flat.view(B, n, n)) # (B, n, n)
                
                # --- 预测矩阵 L ---
                # L 使用与 A 相同的输入 Pairs_A
                L_raw_flat = self.net_L(Pairs_A.view(-1, Pairs_A.shape[-1]))
                L_raw = L_raw_flat.view(B, n, n+1, n+1).squeeze(-1)
                L_mat = torch.zeros(B, n, n+1, n+1, device=device)
                # 取严格上三角部分
                L_mat[:, :, topo['triu_idx_A'][0], topo['triu_idx_A'][1]] = L_raw[:, :, topo['triu_idx_A'][0], topo['triu_idx_A'][1]]
                # 计算对角线 1 到 n 的特征值
                a_pred = L_raw[:, :, topo['diag_idx'], topo['diag_idx']] # (B, n, n)
                a_relu = F.relu(a_pred)
                a_cumsum = torch.cumsum(a_relu, dim=-1) # (B, n, n), 计算累加和 a1, a1+a2, a1+a2+a3...
                epsilon = 1e-6
                diag_vals = 1.0 / (1.0 + a_cumsum + epsilon) # 按照公式计算特征值
                # 将计算好的特征值赋给 L 的对角线
                diag_vals = diag_vals * perc_diag
                L_mat[:, :, topo['diag_idx'], topo['diag_idx']] = diag_vals
                L_mat[:, :, 0, 0] = 1.0 # 强制 [0,0] 为 1
                
                # --- 生成正交矩阵 Q 与细分矩阵 S ---
                B_mat = A - A.transpose(2, 3) 
                Q_flat = torch.matrix_exp(B_mat.view(B*n, n+1, n+1))   
                Q = Q_flat.view(B, n, n+1, n+1)
                S = torch.matmul(Q, torch.matmul(L_mat, Q.transpose(2, 3)))

                s_sum, s_count = sparse_s_loss_sum_count(S, self.sparse_a)
                sparse_loss_sum = sparse_loss_sum + s_sum
                sparse_loss_count += s_count

                V_in_local = torch.zeros(B, n, n+1, 3, device=device)
                V_in_local[:, :, 1:, :] = v1_local 
                V_out_local = torch.matmul(S, V_in_local)
                V_out_local_flat = V_out_local.view(B*n, n+1, 3)
                V_out_global_flat = torch.bmm(V_out_local_flat, local_frames)
                V_out_global = V_out_global_flat.view(B, n, n+1, 3) + pos[v0_idx].view(B, 1, 1, 3)
                
                F_in_local = torch.zeros(B, n, n+1, self.Dout_feature, device=device)
                F_in_local[:, :, 0, :] = updated_v0_feat.unsqueeze(1).expand(B, n, self.Dout_feature)
                F_in_local[:, :, 1:, :] = feat_neighbor
                F_out_local = torch.matmul(S, F_in_local)
                
                gather_idx_V = topo['j_gather'].view(1, n, n, 1).expand(B, n, n, 3)
                V_aligned_flaps = torch.gather(V_out_global[:, :, 1:, :], 2, gather_idx_V)
                V_aligned = torch.cat([V_out_global[:, :, 0:1, :], V_aligned_flaps], dim=2)
                V_avg = V_aligned.mean(dim=1)
                
                gather_idx_F = topo['j_gather'].view(1, n, n, 1).expand(B, n, n, self.Dout_feature)
                F_aligned_flaps = torch.gather(F_out_local[:, :, 1:, :], 2, gather_idx_F)
                F_aligned = torch.cat([F_out_local[:, :, 0:1, :], F_aligned_flaps], dim=2)
                F_avg = F_aligned.mean(dim=1)
                
                new_pos_even[v0_idx] = V_avg[:, 0, :]
                new_feat_even[v0_idx] = F_avg[:, 0, :]
                he_idx_flat = he_idx.flatten()
                new_pos_odd_he[he_idx_flat] = V_avg[:, 1:, :].reshape(-1, 3)
                new_feat_odd_he[he_idx_flat] = F_avg[:, 1:, :].reshape(-1, self.Dout_feature)

            # 一致性损失
            pred_pos_he0 = new_pos_odd_he[0::2]
            pred_pos_he1 = new_pos_odd_he[1::2]
            loss_pos_consist = F.mse_loss(pred_pos_he0, pred_pos_he1)
            consist_losses.append(loss_pos_consist)

            # 池化更新迭代
            pos = torch.cat([new_pos_even, self.halfEdgePool(new_pos_odd_he)], dim=0)
            curr_feat = torch.cat([new_feat_even, self.halfEdgePool(new_feat_odd_he)], dim=0)
            outputs.append(pos)

        total_consist_loss = sum(consist_losses) if consist_losses else 0.0
        sparse_loss = sparse_loss_sum / sparse_loss_count

        return outputs, total_consist_loss, sparse_loss
