import os
import numpy as np
import glob

def read_obj_vertices(file_path):
    vertices = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices)

def write_obj_with_new_vertices(original_file, output_file, new_vertices):
    with open(original_file, 'r') as f_in:
        lines = f_in.readlines()

    v_idx = 0
    with open(output_file, 'w') as f_out:
        for line in lines:
            if line.startswith('v '):
                if v_idx < len(new_vertices):
                    v = new_vertices[v_idx]
                    f_out.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
                    v_idx += 1
                else:
                    f_out.write(line)
            else:
                f_out.write(line)

def normalize_mesh_folder(root_path, original_mesh_name):
    original_path = os.path.join(root_path, original_mesh_name)
    if not os.path.exists(original_path):
        print(f"错误: 找不到原始网格文件 {original_path}")
        return
    print(f"正在计算原始网格基准: {original_mesh_name}...")
    verts = read_obj_vertices(original_path)
    
    if len(verts) == 0:
        print(f"错误: 原始网格 {original_mesh_name} 没有读取到顶点。")
        return

    v_min = verts.min(axis=0)
    v_max = verts.max(axis=0)
    center = (v_max + v_min) / 2.0
    
    max_side = np.max(v_max - v_min)
    scale = 2.0 / max_side if max_side != 0 else 1.0

    print(f"中心点: {center}, 缩放比例: {scale}")

    subd_pattern = os.path.join(root_path, "subd*", "001.obj")
    subd_files = glob.glob(subd_pattern)

    if not subd_files:
        print("未发现匹配的 subd*/001.obj 文件。请检查路径。")
        return

    for file_path in subd_files:
        print(f"正在处理并覆盖: {file_path}")
        current_verts = read_obj_vertices(file_path)
        
        if len(current_verts) == 0:
            print(f"警告: {file_path} 中没有读取到顶点，跳过。")
            continue
            
        normalized_verts = (current_verts - center) * scale
        
        write_obj_with_new_vertices(file_path, file_path, normalized_verts)

    print("--- 所有网格归一化完成，数据写入正常 ---")


if __name__ == "__main__":
    TARGET_DIRECTORY = "./data_meshes/cartoon_elephant_10/" 
    ORIGINAL_MESH = "original.obj"
    
    normalize_mesh_folder(TARGET_DIRECTORY, ORIGINAL_MESH)
