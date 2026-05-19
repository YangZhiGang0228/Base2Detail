import os
import numpy as np
import glob

def read_obj_vertices(file_path):
    """读取 OBJ 文件中的顶点坐标"""
    vertices = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                # 提取 v 之后的三个浮点数
                parts = line.split()
                # 兼容可能存在的多个空格
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices)

def write_obj_with_new_vertices(original_file, output_file, new_vertices):
    """保持原 OBJ 结构（面信息等），仅替换顶点坐标并保存"""
    # 修复核心：先将文件所有行读取到内存中
    # 这样可以安全地覆盖写入自身，避免文件被提前清空为 0KB
    with open(original_file, 'r') as f_in:
        lines = f_in.readlines()

    v_idx = 0
    # 现在再以 'w' 模式打开目标文件进行写入
    with open(output_file, 'w') as f_out:
        for line in lines:
            if line.startswith('v '):
                # 确保不会超出新顶点的索引范围
                if v_idx < len(new_vertices):
                    v = new_vertices[v_idx]
                    f_out.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
                    v_idx += 1
                else:
                    f_out.write(line)
            else:
                f_out.write(line)

def normalize_mesh_folder(root_path, original_mesh_name):
    """
    归一化网格及其细分结果
    :param root_path: 包含原始网格和 subd 文件夹的根目录
    :param original_mesh_name: 原始网格的文件名 (例如 'original.obj')
    """
    original_path = os.path.join(root_path, original_mesh_name)
    if not os.path.exists(original_path):
        print(f"错误: 找不到原始网格文件 {original_path}")
        return

    # 1. 从原始网格计算归一化参数
    print(f"正在计算原始网格基准: {original_mesh_name}...")
    verts = read_obj_vertices(original_path)
    
    if len(verts) == 0:
        print(f"错误: 原始网格 {original_mesh_name} 没有读取到顶点。")
        return

    v_min = verts.min(axis=0)
    v_max = verts.max(axis=0)
    center = (v_max + v_min) / 2.0
    
    # 计算最大跨度，用于缩放到 [-1, 1]
    max_side = np.max(v_max - v_min)
    scale = 2.0 / max_side if max_side != 0 else 1.0

    print(f"中心点: {center}, 缩放比例: {scale}")

    # 3. 遍历并处理 subd* 文件夹下的 001.obj
    subd_pattern = os.path.join(root_path, "subd*", "001.obj")
    subd_files = glob.glob(subd_pattern)

    if not subd_files:
        print("未发现匹配的 subd*/001.obj 文件。请检查路径。")
        return

    for file_path in subd_files:
        print(f"正在处理并覆盖: {file_path}")
        # 读取当前细分网格顶点
        current_verts = read_obj_vertices(file_path)
        
        if len(current_verts) == 0:
            print(f"警告: {file_path} 中没有读取到顶点，跳过。")
            continue
            
        # 应用原始网格的变换参数
        normalized_verts = (current_verts - center) * scale
        
        # 覆盖写入 (现在安全了)
        write_obj_with_new_vertices(file_path, file_path, normalized_verts)

    print("--- 所有网格归一化完成，数据写入正常 ---")


if __name__ == "__main__":
    # --- 配置区域 ---
    # 请将此处改为你的实际文件夹路径
    TARGET_DIRECTORY = "./data_meshes/cartoon_elephant_10/" 
    # 你的原始网格文件名
    ORIGINAL_MESH = "original.obj"
    
    normalize_mesh_folder(TARGET_DIRECTORY, ORIGINAL_MESH)