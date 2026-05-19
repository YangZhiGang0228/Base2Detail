import os
import sys
cwd = os.getcwd()
sys.path.append(cwd + '/pythongptoolbox/')
sys.path.append(cwd + '/torchgptoolbox_nosparse/')

# torch
import torch
import torchgptoolbox_nosparse as tgp

# pygptoolbox
from readOBJ import readOBJ
from writeOBJ import writeOBJ
from findIdx import findIdx
from midPointUpsampling import midPointUpsampling

# python standard 
import numpy as np
import scipy
import scipy.sparse
import glob
import time
import json
import pickle as pickle
import math

def compute_positional_encoding(degree, d_model=16):
    """
    计算一环邻域的相对位置编码 (基于之前的高频傅里叶特征公式)
    degree: 当前中心顶点的度数 (邻居个数)，即公式中的 n
    d_model: 位置编码的总维度大小 (16 对应指数 k 到 7)
    """
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

def compute_loop_eigenvalues(k):
        def get_loop_matrix(n):
            cos_term = 3/8 + 0.25 * np.cos(2 * np.pi / n)
            beta = (5/8 - cos_term**2) / n
            M = torch.zeros((n + 1, n + 1), dtype=torch.float64)
            
            M[0, 0] = 1 - n * beta
            M[0, 1:] = beta
            
            for i in range(1, n + 1):
                M[i, 0] = 3/8  
                M[i, i] = 3/8  
                prev = (i - 2) % n + 1
                nxt = i % n + 1
                M[i, prev] = 1/8
                M[i, nxt] = 1/8
            return M
        m = get_loop_matrix(k)
        return m

def compute_U_matrix(d):
    N = d + 1
    U = torch.zeros((N, N), dtype=torch.float32)
    k = torch.arange(1, N + 1, dtype=torch.float32)
    x = torch.cos((2 * k - 1) / (2 * N) * math.pi)
    U[:, 0] = 1.0      
    if N > 1:
        U[:, 1] = x         
    for m in range(2, N):
        U[:, m] = 2 * x * U[:, m-1] - U[:, m-2]
    return U

class Mesh: 
    def __init__(self, V, F, hfList):
        self.V = V
        self.F = F
        self.hfList = hfList

def processTrainShapes(folder):
    subFolders = os.listdir(folder)
    subFolders = [x for x in subFolders if not (x.startswith('.'))]
    nSubd = len(subFolders) - 1
    objPaths = glob.glob(folder + subFolders[0] + "/*.obj")
    nObjs = len(objPaths)
    paths = []
    for ii in range(nSubd + 1):
        paths.append(folder + 'subd' + str(ii) + '/')
    objFiles = []
    for ii in range(nObjs):
        objFiles.append(str(ii + 1).zfill(3))
    meshes = [None] * nObjs
    for ii in range(nObjs):
        print('process meshes %d / %d' % (ii, nObjs))
        meshes_i = [None] * (nSubd + 1)
        for jj in range(nSubd + 1):
            V, F = tgp.readOBJ(paths[jj] + objFiles[ii] + '.obj')
            _, hfList = computeFlapList(V, F, 1)
            meshes_i[jj] = Mesh(V, F, hfList[0])
        meshes[ii] = list(meshes_i)
    print('num Subd: %d, num meshes: %d' % (nSubd, nObjs))
    return meshes

class TrainMeshes: 
    def __init__(self, folders):
        nShape = len(folders)
        self.meshes = []
        for fIdx in range(nShape):
            meshes = processTrainShapes(folders[fIdx])
            for ii in range(len(meshes)):
                self.meshes.append(meshes[ii])
        self.nM = len(self.meshes)
        self.nS = len(self.meshes[0])
        self.hfList = None
        self.poolMats = None
        self.dofs = None
        self.LCs = None
        self.VNs = None  # 新增法向存储变量
        self.Us = None       
        self.Lambdas = None
        self.groupedLocalData = None

    def getVertexNormals(self):
        """单独计算所有 mesh 的顶点法向"""
        VNs = [None] * self.nM
        for mIdx in range(self.nM):
            V = self.meshes[mIdx][0].V
            F = self.meshes[mIdx][0].F
            VN = tgp.vertexNormals(V, F)
            VNs[mIdx] = VN
        return VNs

    def getInputData(self, mIdx):
        # 拼接全局特征: 坐标 + 法向 + 微分坐标
        input = torch.cat((
            self.meshes[mIdx][0].V,
            self.VNs[mIdx],
            self.LCs[mIdx]),
            dim=1)
        return input

    def getHalfFlap(self):
        HF = [None] * self.nM
        for ii in range(self.nM):
            fifj = [None] * (self.nS - 1)
            for jj in range(self.nS - 1):
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
                poolMat = torch.sparse.FloatTensor(I, val, torch.Size([nV, hfIdx.size(0)]))
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

    def get_U_and_Lambda(self, dofs):
        Us_all = [None] * self.nM
        Lambdas_all = [None] * self.nM
        for mIdx in range(self.nM):
            n_levels = len(dofs[mIdx])
            Us_m = [None] * n_levels
            Lambdas_m = [None] * n_levels
            prev_U_exact = None
            prev_Lambda_exact = None
            for jj in range(n_levels):
                dof = dofs[mIdx][jj].long()
                nV = dof.size(0)
                max_d = dof.max().item()
                U_tensor = torch.zeros(nV, max_d + 1, max_d + 1)
                Lambda_tensor = torch.zeros(nV, max_d + 1, max_d + 1)
                if jj == 0:
                    for i in range(nV):
                        d = dof[i].item()
                        U_tensor[i, :d+1, :d+1] = compute_U_matrix(d)
                        Lambda_tensor[i, :d+1, :d+1] = compute_loop_eigenvalues(d)
                else:
                    nV_prev = dofs[mIdx][jj-1].size(0)
                    max_d_prev = prev_U_exact.size(1) - 1
                    min_max_d = min(max_d, max_d_prev)
                    U_tensor[:nV_prev, :min_max_d+1, :min_max_d+1] = \
                        prev_U_exact[:nV_prev, :min_max_d+1, :min_max_d+1] 
                    Lambda_tensor[:nV_prev, :min_max_d+1, :min_max_d+1] = \
                        prev_Lambda_exact[:nV_prev, :min_max_d+1, :min_max_d+1]
                    for i in range(nV_prev, nV):
                        d = dof[i].item()
                        U_tensor[i, :d+1, :d+1] = compute_U_matrix(d)
                        Lambda_tensor[i, :d+1, :d+1] = compute_loop_eigenvalues(d)
                prev_U_exact = U_tensor.clone()
                prev_Lambda_exact = Lambda_tensor.clone()
                Us_m[jj] = U_tensor
                Lambdas_m[jj] = Lambda_tensor
            Us_all[mIdx] = Us_m
            Lambdas_all[mIdx] = Lambdas_m
        return Us_all, Lambdas_all

    def getGroupedLocalPatches(self, hfList):
        grouped_data = [None] * self.nM
        for mIdx in range(self.nM):
            grouped_data_m = [None] * (self.nS - 1)
            for jj in range(self.nS - 1):
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
                        # 注入相对位置编码
                        deg_dict[degree]['pos_enc'] = compute_positional_encoding(degree, d_model=16)
                    
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

    def computeParameters(self):
        self.hfList = self.getHalfFlap()
        self.poolMats, self.dofs = self.getFlapPool(self.hfList)
        self.LCs = self.getLaplaceCoordinate(self.hfList, self.poolMats, self.dofs)
        self.VNs = self.getVertexNormals() # <--- 新增
        self.Us, self.Lambdas = self.get_U_and_Lambda(self.dofs)
        self.groupedLocalData = self.getGroupedLocalPatches(self.hfList)

    def toDevice(self, device):
        for ii in range(self.nM):
            for jj in range(self.nS):
                self.meshes[ii][jj].V = self.meshes[ii][jj].V.to(device)
                self.meshes[ii][jj].F = self.meshes[ii][jj].F.to(device)
        for ii in range(self.nM):
            self.LCs[ii] = self.LCs[ii].to(device)
            self.VNs[ii] = self.VNs[ii].to(device) # <--- 新增
            for jj in range(self.nS - 1):
                self.hfList[ii][jj] = self.hfList[ii][jj].to(device)
                self.poolMats[ii][jj] = self.poolMats[ii][jj].to(device)
                self.dofs[ii][jj] = self.dofs[ii][jj].to(device)
                self.Us[ii][jj] = self.Us[ii][jj].to(device)
                self.Lambdas[ii][jj] = self.Lambdas[ii][jj].to(device)
                
                deg_dict = self.groupedLocalData[ii][jj]
                for deg in deg_dict:
                    deg_dict[deg]['v0'] = deg_dict[deg]['v0'].to(device)
                    deg_dict[deg]['flap_idx'] = deg_dict[deg]['flap_idx'].to(device)
                    deg_dict[deg]['he_idx'] = deg_dict[deg]['he_idx'].to(device)
                    if 'pos_enc' in deg_dict[deg]:
                        deg_dict[deg]['pos_enc'] = deg_dict[deg]['pos_enc'].to(device) # <--- 新增

def preprocessTestShapes(meshPathList, nSubd=2):
    nObjs = len(meshPathList)
    meshes = [None] * nObjs
    for meshIdx in range(nObjs):
        path = meshPathList[meshIdx]
        V, F = tgp.readOBJ(path)
        FList, hfList = computeFlapList(V, F, nSubd)
        meshes_i = [None] * (nSubd + 1)
        for jj in range(nSubd + 1):
            if jj == 0:
                meshes_i[jj] = Mesh(V, FList[jj], hfList[jj])
            else:
                meshes_i[jj] = Mesh(None, FList[jj], hfList[jj])
        meshes[meshIdx] = list(meshes_i)
    print('num Subd: %d, num meshes: %d' % (nSubd, nObjs))
    return meshes

class TestMeshes:
    def __init__(self, meshPathList, nSubd=2):
        nShape = len(meshPathList)
        self.meshes = preprocessTestShapes(meshPathList, nSubd)
        self.nM = len(self.meshes)
        self.nS = len(self.meshes[0])
        self.hfList = None
        self.poolMats = None
        self.dofs = None
        self.LCs = None
        self.VNs = None # 新增法向存储变量
        self.Us = None       
        self.Lambdas = None
        self.groupedLocalData = None

    def getVertexNormals(self):
        """单独计算所有 mesh 的顶点法向"""
        VNs = [None] * self.nM
        for mIdx in range(self.nM):
            V = self.meshes[mIdx][0].V
            F = self.meshes[mIdx][0].F
            VN = tgp.vertexNormals(V, F)
            VNs[mIdx] = VN
        return VNs

    def getInputData(self, mIdx):
        # 拼接全局特征: 坐标 + 法向 + 微分坐标
        input = torch.cat((
            self.meshes[mIdx][0].V,
            self.VNs[mIdx],
            self.LCs[mIdx]),
            dim=1)
        return input

    def getHalfFlap(self):
        HF = [None] * self.nM
        for ii in range(self.nM):
            fifj = [None] * (self.nS - 1)
            for jj in range(self.nS - 1):
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
                poolMat = torch.sparse.FloatTensor(I, val, torch.Size([nV, hfIdx.size(0)]))
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

    def get_U_and_Lambda(self, dofs):
        Us_all = [None] * self.nM
        Lambdas_all = [None] * self.nM
        for mIdx in range(self.nM):
            n_levels = len(dofs[mIdx])
            Us_m = [None] * n_levels
            Lambdas_m = [None] * n_levels
            prev_U_exact = None
            prev_Lambda_exact = None
            for jj in range(n_levels):
                dof = dofs[mIdx][jj].long()
                nV = dof.size(0)
                max_d = dof.max().item()
                U_tensor = torch.zeros(nV, max_d + 1, max_d + 1)
                Lambda_tensor = torch.zeros(nV, max_d + 1, max_d + 1)
                if jj == 0:
                    for i in range(nV):
                        d = dof[i].item()
                        U_tensor[i, :d+1, :d+1] = compute_U_matrix(d)
                        Lambda_tensor[i, :d+1, :d+1] = compute_loop_eigenvalues(d)
                else:
                    nV_prev = dofs[mIdx][jj-1].size(0)
                    max_d_prev = prev_U_exact.size(1) - 1
                    min_max_d = min(max_d, max_d_prev)
                    U_tensor[:nV_prev, :min_max_d+1, :min_max_d+1] = \
                        prev_U_exact[:nV_prev, :min_max_d+1, :min_max_d+1] 
                    Lambda_tensor[:nV_prev, :min_max_d+1, :min_max_d+1] = \
                        prev_Lambda_exact[:nV_prev, :min_max_d+1, :min_max_d+1]
                    for i in range(nV_prev, nV):
                        d = dof[i].item()
                        U_tensor[i, :d+1, :d+1] = compute_U_matrix(d)
                        Lambda_tensor[i, :d+1, :d+1] = compute_loop_eigenvalues(d)
                prev_U_exact = U_tensor.clone()
                prev_Lambda_exact = Lambda_tensor.clone()
                Us_m[jj] = U_tensor
                Lambdas_m[jj] = Lambda_tensor
            Us_all[mIdx] = Us_m
            Lambdas_all[mIdx] = Lambdas_m
        return Us_all, Lambdas_all

    def getGroupedLocalPatches(self, hfList):
        grouped_data = [None] * self.nM
        for mIdx in range(self.nM):
            grouped_data_m = [None] * (self.nS - 1)
            for jj in range(self.nS - 1):
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
                        # 注入相对位置编码
                        deg_dict[degree]['pos_enc'] = compute_positional_encoding(degree, d_model=16)
                    
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

    def computeParameters(self):
        self.hfList = self.getHalfFlap()
        self.poolMats, self.dofs = self.getFlapPool(self.hfList)
        self.LCs = self.getLaplaceCoordinate(self.hfList, self.poolMats, self.dofs)
        self.VNs = self.getVertexNormals() # <--- 新增
        self.Us, self.Lambdas = self.get_U_and_Lambda(self.dofs)
        self.groupedLocalData = self.getGroupedLocalPatches(self.hfList)

    def toDevice(self, device):
        for ii in range(self.nM):
            for jj in range(self.nS):
                if jj == 0:
                    self.meshes[ii][jj].V = self.meshes[ii][jj].V.to(device)
                self.meshes[ii][jj].F = self.meshes[ii][jj].F.to(device)
        for ii in range(self.nM):
            self.LCs[ii] = self.LCs[ii].to(device)
            self.VNs[ii] = self.VNs[ii].to(device) # <--- 新增
            for jj in range(self.nS - 1):
                self.hfList[ii][jj] = self.hfList[ii][jj].to(device)
                self.poolMats[ii][jj] = self.poolMats[ii][jj].to(device)
                self.dofs[ii][jj] = self.dofs[ii][jj].to(device)
                self.Us[ii][jj] = self.Us[ii][jj].to(device)
                self.Lambdas[ii][jj] = self.Lambdas[ii][jj].to(device)
                
                deg_dict = self.groupedLocalData[ii][jj]
                for deg in deg_dict:
                    deg_dict[deg]['v0'] = deg_dict[deg]['v0'].to(device)
                    deg_dict[deg]['flap_idx'] = deg_dict[deg]['flap_idx'].to(device)
                    deg_dict[deg]['he_idx'] = deg_dict[deg]['he_idx'].to(device)
                    if 'pos_enc' in deg_dict[deg]:
                        deg_dict[deg]['pos_enc'] = deg_dict[deg]['pos_enc'].to(device) # <--- 新增

def computeFlapList(V, F, numSubd=2):
    FList = []
    halfFlapList = []
    for iter in range(numSubd):
        nV = V.size(0)
        VV, FF, S = tgp_midPointUp(V, F, 1)
        rIdx = S._indices()[0, :]
        cIdx = S._indices()[1, :]
        val = S._values()
        cIdx = cIdx[rIdx >= nV]
        val = val[rIdx >= nV]
        rIdx = rIdx[rIdx >= nV]
        assert ((val == 0.5).all())
        rIdx, idx = torch.sort(rIdx)
        cIdx = cIdx[idx]
        rIdx = rIdx[::2]  
        cIdx = cIdx.view(-1, 2)
        flapIdx = torch.zeros(len(rIdx), 4).long()
        for kk in range(len(rIdx)):
            vi = cIdx[kk, 0]
            vj = cIdx[kk, 1]
            adjFi, _ = tgp.findIdx(F, vi)
            adjFj, _ = tgp.findIdx(F, vj)
            adjF = tgp.intersect1d(adjFi, adjFj)
            assert (adjF.size(0) == 2)
            f1 = F[adjF[0], :]
            f2 = F[adjF[1], :]
            f1roll = -tgp.findIdx(f1, vi)[0]
            f2roll = -tgp.findIdx(f2, vi)[0]
            f1_vi = tgp.roll1d(f1, f1roll)
            f2_vi = tgp.roll1d(f2, f2roll)
            if f1_vi[1] == vj:
                f_first = f1_vi  
                f_sec = f2_vi  
            elif f2_vi[1] == vj:
                f_first = f2_vi  
                f_sec = f1_vi  
            flapIdx[kk, :] = torch.tensor([vi, vj, f_first[2], f_sec[1]])
        halfFlapIdx = flapIdx[:, [0, 1, 2, 3, 1, 0, 3, 2]]
        halfFlapIdx = halfFlapIdx.reshape(-1, 4)
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