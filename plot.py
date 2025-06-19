import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

def plot_bandwidth_data(csv_file='results.csv'):
    """
    读取 benchmark 结果并为每个 step size 绘制一张图，
    同时在图上标注出最初和最末的带宽值。
    """
    # 检查CSV文件是否存在
    if not os.path.exists(csv_file):
        print(f"Error: The file '{csv_file}' was not found.")
        print("Please run the ./test.sh script first to generate the data.")
        return

    # 使用 pandas 读取数据
    df = pd.read_csv(csv_file)
    
    # 获取所有独特的 step size
    step_sizes = sorted(df['step'].unique())
    
    print(f"Found data for step sizes: {step_sizes}")

    # 为每个 step_size 创建一张图
    for step in step_sizes:
        # 筛选出当前 step 的数据
        subset = df[df['step'] == step].copy()
        
        # 将带宽从字符串转换为数值，以防万一
        subset['bandwidth'] = pd.to_numeric(subset['bandwidth'])

        # 创建新的图表
        plt.figure(figsize=(12, 8)) # 增加图表高度给标注留出空间
        
        # 绘制折线图
        plt.plot(subset['stride'], subset['bandwidth'], marker='o', linestyle='-', markersize=5, label=f'Read Size = {step}B')
        
        # --- 设置图表样式 ---
        plt.xscale('log', base=2)
        plt.yscale('log', base=2)
        plt.title(f'HBM Memory Throughput vs. Stride (Read Size = {step} Bytes)', fontsize=16)
        plt.xlabel('Stride Interval (Bytes)', fontsize=12)
        plt.ylabel('Achieved Bandwidth (GB/s)', fontsize=12)
        
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.grid(True, which="both", linestyle='--', linewidth=0.5)
        plt.xticks(rotation=45)
        
        # ======================================================
        #  核心修改点：添加最初和最末的带宽值标注
        # ======================================================
        if not subset.empty:
            # 获取第一个数据点
            first_point = subset.iloc[0]
            x1, y1 = first_point['stride'], first_point['bandwidth']
            
            # 添加第一个点的标注
            plt.annotate(
                f'{y1:.1f} GB/s',  # 标注文本
                xy=(x1, y1),      # 标注指向的点
                xytext=(15, 15),  # 文本位置的偏移量
                textcoords='offset points',
                ha='center',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")
            )

            # 获取最后一个数据点
            last_point = subset.iloc[-1]
            x2, y2 = last_point['stride'], last_point['bandwidth']

            # 添加最后一个点的标注
            plt.annotate(
                f'{y2:.1f} GB/s',   # 标注文本
                xy=(x2, y2),       # 标注指向的点
                xytext=(-40, 15),  # 文本位置的偏移量
                textcoords='offset points',
                ha='center',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")
            )

        # 调整布局
        plt.tight_layout()
        
        # 保存图表到文件
        output_filename = f'bandwidth_plot_step_{step}.png'
        plt.savefig(output_filename)
        print(f"Plot saved as '{output_filename}'")
        
        # 关闭当前图表以释放内存
        plt.close()

if __name__ == '__main__':
    plot_bandwidth_data()