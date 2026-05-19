import numpy as np
import trimesh
import os

def normalize_to_box(vertices):
    """
    将网格顶点归一化到 [-1, 1] 的 Bounding Box 内，并居中。
    """
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    
    # 居中
    center = (max_coords + min_coords) / 2.0
    centered = vertices - center
    
    # 等比缩放
    max_range = np.max(max_coords - min_coords)
    if max_range == 0:
        return centered
        
    scale_factor = 2.0 / max_range
    return centered * scale_factor

def fit_mesh_inside_target(target_vertices, source_vertices, margin=0.98):
    """
    将 source_vertices (四面体) 缩放并平移，使其完全位于 target_vertices (球体) 内部。
    
    参数:
    - target_vertices: 目标网格顶点 (应已居中)
    - source_vertices: 需要被缩放的网格顶点
    - margin: 安全边距，0.98 表示四面体最大顶点距离球壳会有 2% 的空隙
    """
    # 1. 计算目标(球体)的最小内半径
    # 即球体顶点到原点的最短距离，保证在这个距离内绝对安全
    target_radii = np.linalg.norm(target_vertices, axis=1)
    target_inner_radius = np.min(target_radii)
    
    # 2. 将源网格(四面体)居中到原点
    min_c = np.min(source_vertices, axis=0)
    max_c = np.max(source_vertices, axis=0)
    source_center = (max_c + min_c) / 2.0
    centered_source = source_vertices - source_center
    
    # 3. 计算源网格(四面体)的最大外半径
    # 即四面体顶点到原点的最远距离
    source_radii = np.linalg.norm(centered_source, axis=1)
    source_outer_radius = np.max(source_radii)
    
    # 4. 计算最终缩放比例并应用
    if source_outer_radius == 0:
        return centered_source
        
    scale_factor = (target_inner_radius * margin) / source_outer_radius
    fitted_source = centered_source * scale_factor
    
    return fitted_source

def process_meshes(sphere_path, tet_path, output_sphere_path, output_tet_path):
    """
    主处理函数
    """
    # 检查文件是否存在
    if not os.path.exists(sphere_path):
        raise FileNotFoundError(f"找不到球体文件: {sphere_path}")
    if not os.path.exists(tet_path):
        raise FileNotFoundError(f"找不到四面体文件: {tet_path}")

    print("加载网格中...")
    sphere_mesh = trimesh.load(sphere_path)
    tet_mesh = trimesh.load(tet_path)

    # 步骤 1: 将球体归一化到 [-1, 1]
    print("正在归一化球体网格...")
    sphere_mesh.vertices = normalize_to_box(sphere_mesh.vertices)
    
    # 步骤 2: 将四面体缩放到球体内部
    print("正在缩放四面体网格以适应球体...")
    tet_mesh.vertices = fit_mesh_inside_target(
        target_vertices=sphere_mesh.vertices,
        source_vertices=tet_mesh.vertices,
        margin=0.95  # 留出5%的空隙，防止完全贴合
    )

    # 验证输出
    sphere_max = np.max(np.linalg.norm(sphere_mesh.vertices, axis=1))
    tet_max = np.max(np.linalg.norm(tet_mesh.vertices, axis=1))
    print(f"\n验证结果:")
    print(f"球体最大半径: {sphere_max:.4f}")
    print(f"四面体最大半径: {tet_max:.4f}")
    print(f"四面体是否在球内: {'是' if tet_max < sphere_max else '否'}")

    # 导出文件
    print(f"\n导出结果到...")
    sphere_mesh.export(output_sphere_path)
    tet_mesh.export(output_tet_path)
    print(f" -> {output_sphere_path}")
    print(f" -> {output_tet_path}")
    print("处理完成！")

if __name__ == "__main__":
    # ==========================================
    # 用户配置区：在这里指定你的文件路径
    # ==========================================
    SPHERE_INPUT_PATH = "./data_meshes/cartoon_elephant_10/original.obj"   # 替换为你的球体模型路径
    TETRAHEDRON_INPUT_PATH = "./data_meshes/cartoon_elephant_10/subd0/001.obj" # 替换为你的四面体模型路径
    
    SPHERE_OUTPUT_PATH = "normalized_sphere.obj"    # 归一化后球体的输出路径
    TETRAHEDRON_OUTPUT_PATH = "fitted_tet.obj"      # 缩放后四面体的输出路径
    
    # 为了演示不报错，如果你没有现成的文件，可以使用 trimesh 自带的生成器测试：
    # 将下方 process_meshes(...) 注释掉，使用测试代码即可。
    
    process_meshes(
        SPHERE_INPUT_PATH, 
        TETRAHEDRON_INPUT_PATH, 
        SPHERE_OUTPUT_PATH, 
        TETRAHEDRON_OUTPUT_PATH
    )