import math

import torch.nn.functional as F
import torch


def aggregate_cyclic_views(aligned, policy="all", view_offset=0, generator=None):
    """Aggregate aligned one-ring predictions without changing their topology."""
    if policy == "all":
        return aligned.mean(dim=1)
    if policy == "fixed":
        return aligned[:, int(view_offset) % aligned.shape[1]]
    if policy == "random":
        batch = torch.arange(aligned.shape[0], device=aligned.device)
        selected = torch.randint(
            aligned.shape[1], (aligned.shape[0],), device=aligned.device,
            generator=generator)
        return aligned[batch, selected]
    raise ValueError("stage1_view_policy must be 'all', 'fixed', or 'random'")


def select_cyclic_views(flaps, policy="all", view_offset=0, generator=None,
                        return_indices=False):
    """Select cyclic one-ring starts before the expensive matrix networks."""
    batch_size, view_count = flaps.shape[:2]
    if policy == "all":
        indices = torch.arange(view_count, device=flaps.device).view(1, -1)
        indices = indices.expand(batch_size, -1)
        selected = flaps
    elif policy == "fixed":
        indices = torch.full(
            (batch_size, 1), int(view_offset) % view_count,
            dtype=torch.long, device=flaps.device)
        selected = flaps[
            torch.arange(batch_size, device=flaps.device), indices[:, 0]
        ].unsqueeze(1)
    elif policy == "random":
        indices = torch.randint(
            view_count, (batch_size, 1), device=flaps.device,
            generator=generator)
        selected = flaps[
            torch.arange(batch_size, device=flaps.device), indices[:, 0]
        ].unsqueeze(1)
    else:
        raise ValueError("stage1_view_policy must be 'all', 'fixed', or 'random'")
    return (selected, indices) if return_indices else selected

class MLP(torch.nn.Module):
    def __init__(self, Din, Dhid, Dout):
        super(MLP, self).__init__()
        self.layerIn = torch.nn.Linear(Din, Dhid[0])
        self.hidden = torch.nn.ModuleList()
        for ii in range(len(Dhid)-1):
            self.hidden.append(torch.nn.Linear(Dhid[ii], Dhid[ii+1]))
        self.layerOut = torch.nn.Linear(Dhid[-1], Dout)
        self.act = torch.nn.SiLU() 

    def forward(self, x):
        x = self.layerIn(x)
        x = self.act(x)
        for ii in range(len(self.hidden)):
            x = self.hidden[ii](x)
            x = self.act(x)
        x = self.layerOut(x)
        return x

    def project_positions(self, positions, context_dim):
        """Weight-dependent projection, shared only within this forward pass."""
        return F.linear(positions, self.layerIn.weight[:, context_dim:])

    def forward_context(self, context, position_hidden):
        """Exactly the original first affine map, without repeating context."""
        width = context.shape[-1]
        common = F.linear(context, self.layerIn.weight[:, :width], self.layerIn.bias)
        x = self.act(common.unsqueeze(-2) + position_hidden)
        for layer in self.hidden:
            x = self.act(layer(x))
        return self.layerOut(x)

def compute_vertex_normals(V, F):
    vec1 = V[F[:, 1], :] - V[F[:, 0], :]
    vec2 = V[F[:, 2], :] - V[F[:, 0], :]
    FN = torch.cross(vec1, vec2, dim=-1)

    VN = torch.zeros_like(V)
    VN.index_add_(0, F[:, 0], FN)
    VN.index_add_(0, F[:, 1], FN)
    VN.index_add_(0, F[:, 2], FN)

    vn_norm = torch.norm(VN, dim=1, keepdim=True)
    VN = VN / (vn_norm + 1e-8)
    return VN

class SubdGeometricMixin(torch.nn.Module):
    def __init__(self):
        super(SubdGeometricMixin, self).__init__()
        self.pool = torch.nn.AvgPool2d((2,1)) 
        self.topo_cache = {}

    def halfEdgePool(self, fhe):
        fhe = fhe.unsqueeze(0).unsqueeze(0)
        fe = self.pool(fhe)
        fe = fe.squeeze(0).squeeze(0)
        return fe

    def v2hf_flapNormalization(self, hf, normalizeFeature=False):
        localFrames = self.flapNormalization(hf[:,:,:3])
        hf_pos = hf[:,:,:3] - hf[:,:1,:3]
        hf_pos = torch.bmm(hf_pos, localFrames.transpose(1, 2))
        hf_feature = hf[:,:,3:]
        if normalizeFeature:
            hf_feature = torch.bmm(hf_feature, localFrames.transpose(1, 2))
        return torch.cat((hf_pos, hf_feature), dim=2), localFrames

    def v2hf(self, fv, hfIdx):
        fv0 = fv[hfIdx[:,0],:].unsqueeze(1) 
        fv1 = fv[hfIdx[:,1],:].unsqueeze(1) 
        fv2 = fv[hfIdx[:,2],:].unsqueeze(1) 
        fv3 = fv[hfIdx[:,3],:].unsqueeze(1) 
        hf = torch.cat((fv0,fv1,fv2,fv3), dim=1) 

        hf_normalize, localFrames = self.v2hf_flapNormalization(hf, False) 
        hf_normalize = hf_normalize.view(hf_normalize.size(0), -1) 
        hf_normalize = hf_normalize[:,3:] 
        return hf_normalize, localFrames
    
    def v2hf_initNet(self, fv, hfIdx):
        fv0 = fv[hfIdx[:,0],:].unsqueeze(1) 
        fv1 = fv[hfIdx[:,1],:].unsqueeze(1) 
        fv2 = fv[hfIdx[:,2],:].unsqueeze(1) 
        fv3 = fv[hfIdx[:,3],:].unsqueeze(1) 
        hf = torch.cat((fv0,fv1,fv2,fv3), dim=1) 

        hf_normalize, localFrames = self.v2hf_flapNormalization(hf, True) 
        hf_normalize = hf_normalize.view(hf_normalize.size(0), -1) 
        hf_normalize = hf_normalize[:,3:] 
        return hf_normalize, localFrames

    def oneRingPool(self, fhe, poolMat, dof):
        fv = torch.spmm(poolMat, fhe)
        fv /= dof.unsqueeze(1) 
        return fv

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
                'triu_idx_L': torch.triu_indices(n+1, n+1, offset=1, device=device),
                'diag_idx': torch.arange(0, n+1, device=device),
                'j_gather': j_gather
            }
        return self.topo_cache[key]


class SubdBaseNet(SubdGeometricMixin):
    def __init__(self, params):
        super(SubdBaseNet, self).__init__()
        self.Dout_feature = params['Dout']
        self.numSubd = params["numSubd"]
        self.group_batch = 4096
        self.PE_dim = int(params['stage1_pe_dim'])
        self.view_policy = params.get("stage1_view_policy", "all")
        self.view_offset = int(params.get("stage1_view_offset", 0))
        # Tiny groups are launch-bound: keep one original batched affine map.
        self.context_reuse_min_views = 512
        
        self.V_feat_dim = 9 + 4 * self.Dout_feature + 2 * self.PE_dim
        
        #self.net_init = MLP(4 * 6 - 3, params['h_initNet'], self.Dout_feature)
        self.net_init = torch.nn.Linear(4 * 6 - 3,self.Dout_feature)
        self.net_feature = MLP(4 * (3 + self.Dout_feature) - 3, params['h_FNet'], self.Dout_feature)
        
        self.net_L_diag = MLP(9 + 4 * self.Dout_feature + self.PE_dim, params['h_matrixNet'], 1)
        self.net_L_triu = MLP(self.V_feat_dim, params['h_matrixNet'], 1)

    def _degree_update(self, pos, curr_feat, v0_idx, flaps, Q_base, view_ids,
                       diag_hidden, triu_hidden, pe):
        flap_idx, selected = select_cyclic_views(
            flaps, self.view_policy, self.view_offset, return_indices=True)
        B, views, n = flap_idx.shape[:3]
        if view_ids.dim() == 1:
            view_ids = view_ids.view(1, n).expand(B, n)
        view_indices = torch.gather(view_ids, 1, selected)
        topo = self._get_topo_indices(n, pos.device)
        reference = pos[flap_idx[:, :, 0, :]]
        frames = self.flapNormalization(reference.reshape(B * views, 4, 3))
        center = pos[v0_idx].view(B, 1, 1, 3)
        neighbors = pos[flap_idx[:, :, :, 1]] - center
        local_neighbors = torch.bmm(
            neighbors.reshape(B * views, n, 3), frames.transpose(1, 2)
        ).view(B, views, n, 3)
        # Only the first half-flap supplies the 9D geometric context.
        reference_local = torch.bmm(
            (reference[:, :, 1:, :] - center).reshape(B * views, 3, 3),
            frames.transpose(1, 2)).view(B, views, 9)
        features = curr_feat[flap_idx[:, :, 0, :]].reshape(
            B, views, 4 * self.Dout_feature)
        context = torch.cat((reference_local, features), dim=-1)
        r, c = topo['triu_idx_L']
        if diag_hidden is None or B * views < self.context_reuse_min_views:
            diagonal_input = torch.cat((context.unsqueeze(2).expand(B, views, n+1, -1),
                                       pe.expand(B, views, n+1, -1)), dim=-1)
            pair_pe = torch.cat((pe[r], pe[c]), dim=-1)
            upper_input = torch.cat((context.unsqueeze(2).expand(B, views, len(r), -1),
                                    pair_pe.expand(B, views, len(r), -1)), dim=-1)
            raw_s = self.net_L_diag(diagonal_input.reshape(-1, diagonal_input.shape[-1])).view(B, views, n+1)
            upper = self.net_L_triu(upper_input.reshape(-1, upper_input.shape[-1])).view(B, views, len(r))
        else:
            raw_s = self.net_L_diag.forward_context(context, diag_hidden).squeeze(-1)
            upper = self.net_L_triu.forward_context(context, triu_hidden).squeeze(-1)
        s = F.softplus(raw_s)
        sums = s.flip(-1).cumsum(-1).flip(-1)
        lambdas = torch.cat((torch.ones_like(sums[:, :, :1]),
                             sums[:, :, 1:] / (sums[:, :, :1] + 1e-8)), dim=-1)
        L = pos.new_zeros(B, views, n + 1, n + 1)
        L[:, :, r, c] = upper
        L[:, :, topo['diag_idx'], topo['diag_idx']] = lambdas
        Q = Q_base.view(1, 1, n + 1, n + 1)
        local = torch.cat((torch.zeros_like(local_neighbors[:, :, :1]),
                           local_neighbors), dim=2)
        # Q L Q^T X: associate with the three coordinate columns, not Q^T.
        transformed = torch.matmul(Q, torch.matmul(L, torch.matmul(Q.transpose(-1, -2), local)))
        global_out = torch.bmm(
            transformed.reshape(B * views, n + 1, 3), frames
        ).view(B, views, n + 1, 3) + center
        gather = topo['j_gather'][view_indices].unsqueeze(-1).expand(B, views, n, 3)
        aligned = torch.gather(global_out[:, :, 1:, :], 2, gather)
        return torch.cat((global_out[:, :, :1, :], aligned), dim=2).mean(dim=1)

    def forward(self, fv, mIdx, HFs, poolMats, DOFs, groupedLocalData,
                return_features=True):
        pos = fv[:, :3]
        fhf, _ = self.v2hf_initNet(fv, HFs[mIdx][0])
        curr_feat = self.oneRingPool(self.net_init(fhf), poolMats[mIdx][0], DOFs[mIdx][0])
        outputs = [pos]
        for level in range(self.numSubd):
            nV, nHE = DOFs[mIdx][level].size(0), HFs[mIdx][level].size(0)
            even, odd = pos.new_zeros(nV, 3), pos.new_zeros(nHE, 3)
            need_features = return_features or level + 1 < self.numSubd
            next_features = None
            if need_features:
                fhf, _ = self.v2hf(torch.cat((pos, curr_feat), dim=1), HFs[mIdx][level])
                updated = self.net_feature(fhf)
                next_features = torch.cat((
                    self.oneRingPool(updated, poolMats[mIdx][level], DOFs[mIdx][level]),
                    self.halfEdgePool(updated)), dim=0)
            for degree, data in groupedLocalData[mIdx][level].items():
                topo = self._get_topo_indices(degree, pos.device)
                pe = torch.cat((data['pos_enc'].new_zeros(1, self.PE_dim),
                                data['pos_enc']), dim=0)
                r, c = topo['triu_idx_L']
                context_dim = 9 + 4 * self.Dout_feature
                # These projections depend on trainable weights: rebuild each forward.
                views = degree if self.view_policy == 'all' else 1
                diag_hidden, triu_hidden = None, None
                if len(data['v0']) * views >= self.context_reuse_min_views:
                    diag_hidden = self.net_L_diag.project_positions(pe, context_dim)
                    triu_hidden = self.net_L_triu.project_positions(
                        torch.cat((pe[r], pe[c]), dim=-1), context_dim)
                batch = getattr(self, 'group_batch', None) or len(data['v0'])
                view_ids = data.get('view_ids', torch.arange(degree, device=pos.device))
                for start in range(0, len(data['v0']), batch):
                    stop = start + batch
                    ids = view_ids if view_ids.dim() == 1 else view_ids[start:stop]
                    args = (pos, curr_feat, data['v0'][start:stop],
                            data['flap_idx'][start:stop], data['Q'], ids,
                            diag_hidden, triu_hidden, pe)
                    if torch.is_grad_enabled() and getattr(self, 'use_checkpoint', False):
                        from torch.utils.checkpoint import checkpoint
                        average = checkpoint(self._degree_update, *args, use_reentrant=False)
                    else:
                        average = self._degree_update(*args)
                    even[data['v0'][start:stop]] = average[:, 0, :]
                    odd[data['he_idx'][start:stop].flatten()] = average[:, 1:, :].reshape(-1, 3)
            pos = torch.cat((even, self.halfEdgePool(odd)), dim=0)
            curr_feat = next_features
            outputs.append(pos)
        return outputs, curr_feat

class SharedVertexMLP(torch.nn.Module):

    def __init__(self, widths):
        super(SharedVertexMLP, self).__init__()
        self.widths = [int(width) for width in widths]
        widths = self.widths
        self.layer1 = torch.nn.Linear(widths[0], widths[1])
        self.layer2 = torch.nn.Linear(widths[1], widths[2])
        self.layer3 = torch.nn.Linear(widths[2], widths[3])
        self.output = torch.nn.Linear(widths[3], widths[4])
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.layer1.weight.uniform_(
                -1.0 / self.layer1.in_features, 1.0 / self.layer1.in_features)
            self.layer1.bias.zero_()
            for layer in (self.layer2, self.layer3):
                bound = math.sqrt(6.0 / layer.in_features)
                layer.weight.uniform_(-bound, bound)
                layer.bias.zero_()
            self.output.weight.zero_()
            self.output.bias.zero_()

    def forward(self, features):
        hidden = torch.sin(30.0 * self.layer1(features))
        hidden = hidden + torch.sin(self.layer2(hidden))
        hidden = torch.sin(self.layer3(hidden))
        return self.output(hidden)


class SubdDispNet(torch.nn.Module):

    def __init__(self, params):
        super(SubdDispNet, self).__init__()
        self.feature_mode = params.get('disp_feature_mode', 'fourier')
        self.n_anchor = int(params['disp_n_anchor'])
        self.anchor_pe_dim = int(params['disp_anchor_pe_dim'])
        self.frequency_count = self.anchor_pe_dim // 2
        if self.feature_mode == 'raw_distance':
            self.input_width = self.n_anchor
        elif self.feature_mode == 'fourier':
            if self.anchor_pe_dim <= 0 or self.anchor_pe_dim % 2:
                raise ValueError('disp_anchor_pe_dim must be a positive even number')
            self.input_width = self.n_anchor * self.anchor_pe_dim
        else:
            raise ValueError('unsupported disp_feature_mode: ' + self.feature_mode)
        self.register_buffer(
            'anchor_idx', torch.zeros(self.n_anchor, dtype=torch.long))
        self.register_buffer('anchor_ready', torch.zeros(1, dtype=torch.bool))
        self.net_disp = SharedVertexMLP(
            [self.input_width] + list(params['h_disp']) + [1])
        self.output_scale = 1.0

    def select_anchors(self, pos):
        distance_to_center = (pos - pos.mean(dim=0, keepdim=True)).norm(dim=1)
        anchors = [int(torch.argmax(distance_to_center))]
        nearest = (pos - pos[anchors[0]]).norm(dim=1)
        for _ in range(1, self.n_anchor):
            nearest[anchors] = -1.0
            picked = int(torch.argmax(nearest))
            anchors.append(picked)
            nearest = torch.minimum(nearest, (pos - pos[picked]).norm(dim=1))
        return torch.tensor(anchors, dtype=torch.long, device=pos.device)

    def anchor_features(self, pos):
        if not bool(self.anchor_ready.item()):
            raise ValueError("Initialize or load the Stage 2 anchor indices before using the model.")
        distance = (pos[:, None, :] - pos[self.anchor_idx][None, :, :]).norm(dim=2)
        distance = distance - distance.mean(dim=0, keepdim=True)
        distance = distance / distance.std(
            dim=0, keepdim=True, unbiased=False).clamp_min(1e-8)
        if self.feature_mode == 'raw_distance':
            return distance
        frequency = 2.0 ** torch.arange(
            self.frequency_count, device=pos.device, dtype=torch.float64)
        phase = math.pi * distance.to(torch.float64)[:, :, None] * frequency
        return torch.cat((torch.sin(phase), torch.cos(phase)), dim=2).reshape(
            len(pos), self.input_width).to(pos.dtype)

    def forward(self, pos, F=None, **kwargs):
        normal = compute_vertex_normals(pos, F)
        displacement = self.net_disp(self.anchor_features(pos)) * self.output_scale
        return pos + displacement * normal
