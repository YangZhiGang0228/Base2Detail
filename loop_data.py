import numpy as np
import os
import sys
import math

def read_obj(filename):
    """简易 OBJ 读取器"""
    V, F = [], []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v '):
                V.append([float(x) for x in line.strip().split()[1:]])
            elif line.startswith('f '):
                # 处理 f 1/2/3 或者 f 1 的情况，索引从0开始
                parts = line.strip().split()[1:]
                F.append([int(x.split('/')[0]) - 1 for x in parts])
    return np.array(V), np.array(F)

def write_obj(filename, V, F):
    """简易 OBJ 写入器"""
    with open(filename, 'w') as f:
        for v in V:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in F:
            # 索引恢复为从1开始
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def standard_loop_subdivision_step(V, F):
    """执行单次标准 Loop 细分"""
    nV = len(V)
    nF = len(F)

    # 1. 构建拓扑：生成半边及唯一边
    # 按照 [v0, v1], [v1, v2], [v2, v0] 顺序排列
    hE = np.vstack((F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]))
    hE_sorted = np.sort(hE, axis=1) # 排序以找到相同的边
    
    # 类似 igl::unique_rows
    E, uniqueIdx = np.unique(hE_sorted, axis=0, return_inverse=True)
    nE = len(E)

    # 2. 构造新面片索引
    # 生成新顶点（边中点）的索引，对应原先的 e0, e1, e2
    e0 = nV + uniqueIdx[0 : nF]
    e1 = nV + uniqueIdx[nF : 2*nF]
    e2 = nV + uniqueIdx[2*nF : 3*nF]

    NF = np.empty((nF * 4, 3), dtype=int)
    NF[0*nF : 1*nF] = np.column_stack((F[:, 0], e0, e2))
    NF[1*nF : 2*nF] = np.column_stack((F[:, 1], e1, e0))
    NF[2*nF : 3*nF] = np.column_stack((F[:, 2], e2, e1))
    NF[3*nF : 4*nF] = np.column_stack((e0, e1, e2))

    # 3. 几何更新：提取邻接关系用于 Standard Loop 加权
    E2Opp = [[] for _ in range(nE)]
    for r in range(3 * nF):
        e = uniqueIdx[r]
        f = r % nF
        ii = r // nF
        opp_v_idx = F[f, (ii + 2) % 3] # 找到相对该边的第三个顶点
        E2Opp[e].append(opp_v_idx)

    V2V = [[] for _ in range(nV)]
    V_is_boundary = [False] * nV
    V2V_boundary = [[] for _ in range(nV)]

    for e_idx in range(nE):
        v0, v1 = E[e_idx]
        V2V[v0].append(v1)
        V2V[v1].append(v0)
        
        # 边界边 (只有1个邻接面)
        if len(E2Opp[e_idx]) == 1:
            V_is_boundary[v0] = True
            V_is_boundary[v1] = True
            V2V_boundary[v0].append(v1)
            V2V_boundary[v1].append(v0)

    NV = np.zeros((nV + nE, 3))

    # [规则 A]: 计算奇顶点 (新生成的边顶点)
    for e_idx in range(nE):
        v0, v1 = E[e_idx]
        if len(E2Opp[e_idx]) == 2:
            # 内部边: 3/8 * (v0 + v1) + 1/8 * (opp1 + opp2)
            NV[nV + e_idx] = 0.375 * (V[v0] + V[v1]) + 0.125 * (V[E2Opp[e_idx][0]] + V[E2Opp[e_idx][1]])
        else:
            # 边界边: 1/2 * (v0 + v1)
            NV[nV + e_idx] = 0.5 * (V[v0] + V[v1])

    # [规则 B]: 计算偶顶点 (被平滑的原始顶点)
    for v in range(nV):
        if V_is_boundary[v]:
            if len(V2V_boundary[v]) == 2:
                # 标准边界点规则: 3/4 * v + 1/8 * (bv1 + bv2)
                NV[v] = 0.75 * V[v] + 0.125 * (V[V2V_boundary[v][0]] + V[V2V_boundary[v][1]])
            else:
                # 非流形或孤立点容错兜底
                NV[v] = V[v]
        else:
            n = len(V2V[v])
            if n == 0:
                NV[v] = V[v]
                continue
            
            cos_val = math.cos(2.0 * math.pi / n)
            alpha = 0.375 + 0.25 * cos_val
            beta = (1.0 / n) * (0.625 - alpha * alpha)

            neighbor_sum = np.sum(V[V2V[v]], axis=0)
            NV[v] = (1.0 - n * beta) * V[v] + beta * neighbor_sum

    return NV, NF

def loop_subdivide_and_save(mesh_path, num_subdivs, output_dir):
    """控制细分迭代与文件输出的主接口"""
    print(f"Loading mesh from: {mesh_path}")
    VO, FO = read_obj(mesh_path)
    
    basename = os.path.splitext(os.path.basename(mesh_path))[0]
    
    currV, currF = VO, FO
    
    # 创建目录并保存第0层
    subd0_dir = os.path.join(output_dir, "subd0")
    os.makedirs(subd0_dir, exist_ok=True)
    out0 = os.path.join(subd0_dir, f"{basename}_level_000.obj")
    write_obj(out0, currV, currF)
    print(f"Saved: {out0} (Vertices: {len(currV)}, Faces: {len(currF)})")

    # 迭代细分
    for i in range(1, num_subdivs + 1):
        nextV, nextF = standard_loop_subdivision_step(currV, currF)
        currV, currF = nextV, nextF
        
        subd_dir = os.path.join(output_dir, f"subd{i}")
        os.makedirs(subd_dir, exist_ok=True)
        out_name = os.path.join(subd_dir, f"{basename}_level_{i:03d}.obj")
        
        write_obj(out_name, currV, currF)
        print(f"Saved: {out_name} (Vertices: {len(currV)}, Faces: {len(currF)})")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python standard_loop_subdiv.py [mesh_path] [num_subdivs] [output_dir]")
        sys.exit(1)

    mesh_path = sys.argv[1]
    num_subdivs = int(sys.argv[2])
    output_dir = sys.argv[3]

    loop_subdivide_and_save(mesh_path, num_subdivs, output_dir)
    print("Subdivision completed successfully!")