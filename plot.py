import pandas as pd
import matplotlib.pyplot as plt

# 您的文件列表
files = [
    './box_loss/neuralsubdivision.txt',
    './box_loss/Lweight_everyelement.txt', # 如果本地是小写，请改为 neuralsubdivision.txt
    './box_loss/valuebylength.txt'
]

# 指定每条曲线的颜色，保持对比一致性
color_map = {
    './box_loss/neuralsubdivision.txt': 'green',
    './box_loss/Lweight_everyelement.txt': 'blue',
    './box_loss/valuebylength.txt': 'orange'
    
}

plt.figure(figsize=(12, 7))

for file in files:
    try:
        # 读取文件，假设每行只有一个浮点数
        df = pd.read_csv(file, header=None, names=['loss'])
        
        # 统一截取前10000步的数据（如果您想画全部数据，请去掉 [:10000]）
        y = df['loss'].values[:10000]
        
        # 直接绘制原始 loss 曲线，不做平滑
        # 适度设置 alpha 防止线条过度堆叠导致无法分辨
        plt.plot(y, label=file.replace('.txt', ''), color=color_map.get(file), linewidth=1, alpha=0.85)
        
    except Exception as e:
        print(f"无法读取文件 {file}: {e}")

# 设置Y轴为对数坐标系，以便更好地观察极小值差距
plt.yscale('log')

# 设置图表标题和坐标轴标签
plt.xlabel('Iteration')
plt.ylabel('Loss (Log Scale)')
plt.title('Raw Training Loss Comparison')

# 设置图例位置（放在图表外部右侧防遮挡）
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

# 开启主次网格线
plt.grid(True, which="major", ls="-", alpha=0.3)
plt.grid(True, which="minor", ls="--", alpha=0.1)

# 自动调整布局并显示/保存
plt.tight_layout()
# plt.savefig('raw_loss_curves.png', dpi=300) # 若需保存图片请取消此行注释
plt.show()