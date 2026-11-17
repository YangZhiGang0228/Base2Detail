import math
from pathlib import Path

import numpy as np
import torch

from tools.midPointUpsampling import midPointUpsampling
from tools.readOBJ import readOBJ

def compute_positional_encoding(degree, d_model):
    if d_model < 2 or d_model % 2:
        raise ValueError('position-encoding dimension must be positive and even')
    pe = torch.zeros(degree, d_model)
    i_idx = torch.arange(0, degree, dtype=torch.float)
    base_angle = (i_idx + 1.0) * 2.0 * math.pi / degree
    base_angle = base_angle.unsqueeze(1) 
    
    k = torch.arange(0, d_model // 2, dtype=torch.float).unsqueeze(0)
    freqs = torch.pow(2.0, k)
    angles = base_angle * freqs 
    
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe

def compute_Q_matrix(n):
    Q = torch.zeros((n+1, n+1), dtype=torch.float32)
    Q[:, 0] = 1.0 / math.sqrt(n+1)
    Q[0, 1] = math.sqrt(n / (n+1))
    Q[1:, 1] = -1.0 / math.sqrt(n * (n+1))
    
    col = 2
    for k in range(1, n // 2 + 1):
        if k == n / 2 and n % 2 == 0:
            for j in range(n):
                Q[j+1, col] = (1.0 / math.sqrt(n)) * math.cos(math.pi * j)
            col += 1
        else:
            for j in range(n):
                Q[j+1, col] = math.sqrt(2.0 / n) * math.cos(2 * math.pi * k * j / n)
            col += 1
            for j in range(n):
                Q[j+1, col] = math.sqrt(2.0 / n) * math.sin(2 * math.pi * k * j / n)
            col += 1
    return Q

class Mesh: 
    def __init__(self, V, F, hfList):
        self.V = V
        self.F = F
        self.hfList = hfList

def processTrainShapes(folder):
    folder = Path(folder)
    levels = sorted(
        int(path.name[4:]) for path in folder.iterdir()
        if path.is_dir() and path.name.startswith("subd") and path.name[4:].isdigit()
    )
    if len(levels) < 2 or levels != list(range(levels[-1] + 1)):
        raise ValueError("Expected contiguous subdivision folders starting with subd0 and subd1.")

    coarse_files = list((folder / "subd0").glob("*.obj"))
    if len(coarse_files) != 1:
        raise ValueError(f"Expected exactly one coarse OBJ in {folder / 'subd0'}.")
    name = coarse_files[0].name
    reference_path = folder / "original" / name
    if not reference_path.is_file():
        raise FileNotFoundError(f"Missing original reference mesh: {reference_path}")

    vertices, faces = readOBJ(reference_path)
    maximum = vertices.max(dim=0).values
    minimum = vertices.min(dim=0).values
    center = (maximum + minimum) / 2.0
    scale = (maximum - minimum).max() / 2.0
    if scale == 0:
        scale = 1e-5
    normalization = {"center": center, "scale": scale}
    reference = Mesh((vertices - center) / scale, faces, None)

    hierarchy = []
    for level in levels:
        mesh_path = folder / f"subd{level}" / name
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Missing subdivision level: {mesh_path}")
        vertices, faces = readOBJ(mesh_path)
        vertices = (vertices - center) / scale
        _, half_flaps = computeFlapList(vertices, faces, 0)
        hierarchy.append(Mesh(vertices, faces, half_flaps[0]))

    print(f"Prepared {name} with {levels[-1]} subdivision levels.")
    return [hierarchy], [normalization], [reference]


class MeshHierarchy:
    def __init__(self, meshes, transforms, orig_meshes):
        self.meshes = meshes
        self.transforms = transforms
        self.orig_meshes = orig_meshes
        self.nM = len(self.meshes)
        self.nS = len(self.meshes[0])
        self.hfList = None
        self.poolMats = None
        self.dofs = None
        self.LCs = None
        #self.VNs = None  
        self.groupedLocalData = None

    def getInputData(self, mIdx):
        input = torch.cat((
            self.meshes[mIdx][0].V,
            #self.VNs[mIdx],
            self.LCs[mIdx]),
            dim=1)
        return input

    def getHalfFlap(self):
        HF = [None] * self.nM
        for ii in range(self.nM):
            fifj = [None] * (self.nS)
            for jj in range(self.nS):
                idx = self.meshes[ii][jj].hfList[:, [0, 1, 2, 3]]
                fifj[jj] = idx.reshape(-1, 4)
            HF[ii] = list(fifj)
        return HF

    def getFlapPool(self, HF):
        nM = len(HF)
        nS = len(HF[0])
        poolFlap = [None] * nM
        dof = [None] * nM
        for ii in range(nM):
            poolFlap_ij = [None] * nS
            dof_ij = [None] * nS
            for jj in range(nS):
                hfIdx = HF[ii][jj]
                nV = hfIdx[:, 0].max() + 1
                rIdx = hfIdx[:, 0]
                cIdx = torch.arange(hfIdx.size(0))
                I = torch.cat([rIdx, cIdx], 0).reshape(2, -1)
                val = torch.ones(hfIdx.size(0))
                poolMat = torch.sparse_coo_tensor(I, val, torch.Size([nV, hfIdx.size(0)]), dtype=val.dtype, device=val.device)
                rowSum = torch.sparse.sum(poolMat, dim=1).to_dense()
                poolFlap_ij[jj] = poolMat
                dof_ij[jj] = rowSum
            poolFlap[ii] = list(poolFlap_ij)
            dof[ii] = list(dof_ij)
        return poolFlap, dof

    def getLaplaceCoordinate(self, hfList, poolMats, dofs):
        LC = [None] * self.nM
        for mIdx in range(self.nM):
            V = self.meshes[mIdx][0].V
            HF = hfList[mIdx][0]
            poolMat = poolMats[mIdx][0]
            dof = dofs[mIdx][0]
            dV_he = V[HF[:, 0], :] - V[HF[:, 1], :]
            dV_v = torch.spmm(poolMat, dV_he)
            dV_v /= dof.unsqueeze(1)
            LC[mIdx] = dV_v
        return LC

    def getGroupedLocalPatches(self, hfList, pe_dim):
        grouped_data = [None] * self.nM
        for mIdx in range(self.nM):
            grouped_data_m = [None] * (self.nS)
            for jj in range(self.nS):
                HF = hfList[mIdx][jj]
                nV = HF[:, 0].max().item() + 1
                v_flaps_idx = {i: [] for i in range(nV)}
                for i in range(HF.size(0)):
                    v0 = HF[i, 0].item()
                    v_flaps_idx[v0].append(i) 
                
                deg_dict = {}
                for v0, f_indices in v_flaps_idx.items():
                    degree = len(f_indices)
                    if degree == 0:
                        continue
                    
                    flaps = [HF[idx] for idx in f_indices] 
                    v1_to_idx = {flaps[k][1].item(): f_indices[k] for k in range(degree)}
                    v1_to_flap = {flaps[k][1].item(): flaps[k] for k in range(degree)}
                    
                    ordered_idx = [f_indices[0]]
                    ordered_flaps = [flaps[0]]
                    curr_f = flaps[0] 
                    
                    while len(ordered_idx) < degree:
                        v2, v3 = curr_f[2].item(), curr_f[3].item() 
                        ordered_flaps.append(v1_to_flap[v2])
                        ordered_idx.append(v1_to_idx[v2]) 
                        curr_f = ordered_flaps[-1]
                    
                    flap_indices_all_views = torch.zeros((degree, degree, 4), dtype=torch.long) 
                    for k in range(degree):
                        for j in range(degree):
                            curr_j = (k + j) % degree
                            flap_indices_all_views[k, j] = ordered_flaps[curr_j]
                    
                    he_indices = torch.tensor(ordered_idx, dtype=torch.long)
                    
                    if degree not in deg_dict:
                        deg_dict[degree] = {'v0': [], 'flap_idx': [], 'he_idx': []}
                        deg_dict[degree]['pos_enc'] = compute_positional_encoding(degree, pe_dim)
                        deg_dict[degree]['Q'] = compute_Q_matrix(degree)
                    
                    deg_dict[degree]['v0'].append(v0)
                    deg_dict[degree]['flap_idx'].append(flap_indices_all_views)
                    deg_dict[degree]['he_idx'].append(he_indices)
                
                for deg in deg_dict:
                    deg_dict[deg]['v0'] = torch.tensor(deg_dict[deg]['v0'], dtype=torch.long)
                    deg_dict[deg]['flap_idx'] = torch.stack(deg_dict[deg]['flap_idx'])
                    deg_dict[deg]['he_idx'] = torch.stack(deg_dict[deg]['he_idx'])
                
                grouped_data_m[jj] = deg_dict
            grouped_data[mIdx] = grouped_data_m
        return grouped_data

    def computeParameters(self, pe_dim):
        self.hfList = self.getHalfFlap()
        self.poolMats, self.dofs = self.getFlapPool(self.hfList)
        self.LCs = self.getLaplaceCoordinate(self.hfList, self.poolMats, self.dofs)
        #self.VNs = self.getVertexNormals() 
        self.groupedLocalData = self.getGroupedLocalPatches(self.hfList, pe_dim)

    def toDevice(self, device):
        for original in self.orig_meshes:
            original.V = original.V.to(device)
            original.F = original.F.to(device)

        for ii in range(self.nM):
            for jj in range(self.nS):
                if jj == 0:
                    self.meshes[ii][jj].V = self.meshes[ii][jj].V.to(device)
                self.meshes[ii][jj].F = self.meshes[ii][jj].F.to(device)
        for ii in range(self.nM):
            self.LCs[ii] = self.LCs[ii].to(device)
            for jj in range(self.nS):
                self.hfList[ii][jj] = self.hfList[ii][jj].to(device)
                self.poolMats[ii][jj] = self.poolMats[ii][jj].to(device)
                self.dofs[ii][jj] = self.dofs[ii][jj].to(device)
                deg_dict = self.groupedLocalData[ii][jj]
                for deg in deg_dict:
                    deg_dict[deg]['v0'] = deg_dict[deg]['v0'].to(device)
                    deg_dict[deg]['flap_idx'] = deg_dict[deg]['flap_idx'].to(device)
                    deg_dict[deg]['he_idx'] = deg_dict[deg]['he_idx'].to(device)
                    if 'pos_enc' in deg_dict[deg]:
                        deg_dict[deg]['pos_enc'] = deg_dict[deg]['pos_enc'].to(device)
                    if 'Q' in deg_dict[deg]:
                        deg_dict[deg]['Q'] = deg_dict[deg]['Q'].to(device)

class TrainMeshes(MeshHierarchy):
    def __init__(self, folder):
        super().__init__(*processTrainShapes(folder))


def load_coarse_mesh(path, depth, normalization):
    """Build the decoding hierarchy from a coarse OBJ and its saved transform."""
    vertices, faces = readOBJ(path)
    vertices = (vertices - normalization["center"]) / normalization["scale"]
    faces_by_level, flaps_by_level = computeFlapList(vertices, faces, depth)

    hierarchy = [Mesh(vertices, faces_by_level[0], flaps_by_level[0])]
    for level in range(1, depth + 1):
        hierarchy.append(Mesh(None, faces_by_level[level], flaps_by_level[level]))
    return MeshHierarchy([hierarchy], [normalization], [])


def computeFlapList(V, F, numSubd=2):
    FList = []
    halfFlapList = []
    
    for iter in range(numSubd+1):
        nV = V.size(0)
        VV, FF, S = tgp_midPointUp(V, F, 1)
        
        rIdx = S._indices()[0, :]
        cIdx = S._indices()[1, :]
        val = S._values()
        
        mask = rIdx >= nV
        cIdx = cIdx[mask]
        val = val[mask]
        rIdx = rIdx[mask]
        assert ((val == 0.5).all())
        
        rIdx, idx = torch.sort(rIdx)
        cIdx = cIdx[idx]
        rIdx = rIdx[::2]  
        cIdx = cIdx.view(-1, 2)
        
        # ==========================================================
        # ==========================================================
        
        he0 = F[:, [0, 1]]
        he1 = F[:, [1, 2]]
        he2 = F[:, [2, 0]]
        HE = torch.cat([he0, he1, he2], dim=0) # Shape: (3*nF, 2)
        
        op0 = F[:, 2]
        op1 = F[:, 0]
        op2 = F[:, 1]
        opV = torch.cat([op0, op1, op2], dim=0) # Shape: (3*nF,)
        
        N = int(VV.size(0)) + 1 
        he_keys = HE[:, 0].long() * N + HE[:, 1].long()
        
        sorted_idx = torch.argsort(he_keys)
        sorted_he_keys = he_keys[sorted_idx]
        sorted_opV = opV[sorted_idx]
        
        vi = cIdx[:, 0].long()
        vj = cIdx[:, 1].long()
        
        c_keys_fwd = vi * N + vj
        c_keys_bwd = vj * N + vi
        
        pos_fwd = torch.searchsorted(sorted_he_keys, c_keys_fwd)
        pos_bwd = torch.searchsorted(sorted_he_keys, c_keys_bwd)
        
        v_first = sorted_opV[pos_fwd]
        v_sec = sorted_opV[pos_bwd]
        
        flapIdx = torch.stack([vi, vj, v_first, v_sec], dim=1)
        halfFlapIdx = flapIdx[:, [0, 1, 2, 3, 1, 0, 3, 2]].reshape(-1, 4)
        
        # ==========================================================
        # ==========================================================
        
        FList.append(F)
        halfFlapList.append(halfFlapIdx)
        V = VV
        F = FF
        
    FList.append(F)
    halfFlapList.append(None)
    return FList, halfFlapList

def tgp_midPointUp(V, F, subdIter=1):
    Vnp = V.data.numpy()
    Fnp = F.data.numpy()
    VVnp, FFnp, SSnp = midPointUpsampling(Vnp, Fnp, subdIter)
    VV = torch.from_numpy(VVnp).float()
    FF = torch.from_numpy(FFnp).long()
    SSnp = SSnp.tocoo()
    values = SSnp.data
    indices = np.vstack((SSnp.row, SSnp.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = SSnp.shape
    SS = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return VV, FF, SS

def random3DRotation():
    theta_x = torch.rand(1) * 2 * np.pi
    sinx = torch.sin(theta_x)
    cosx = torch.cos(theta_x)
    Rx = torch.tensor([[1., 0., 0.],
                       [0., cosx, -sinx],
                       [0., sinx, cosx]])
    theta_y = torch.rand(1) * 2 * np.pi
    siny = torch.sin(theta_y)
    cosy = torch.cos(theta_y)
    Ry = torch.tensor([[cosy, 0., siny],
                       [0., 1., 0.],
                       [-siny, 0., cosy]])
    theta_z = torch.rand(1) * 2 * np.pi
    sinz = torch.sin(theta_z)
    cosz = torch.cos(theta_z)
    Rz = torch.tensor([[cosz, -sinz, 0.],
                       [sinz, cosz, 0.],
                       [0., 0., 1.]])
    return Rx.mm(Ry).mm(Rz)
