import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import seaborn as sns

def plot_bandwidth_data_new(csv_file='results_new.csv'):
    """
    读取新的 benchmark 结果 (包含 reads_per_thread),
    并为每个 step size 绘制一张包含多条曲线的对比图。
    """
    # 检查CSV文件是否存在
    if not os.path.exists(csv_file):
        print(f"Error: The file '{csv_file}' was not found.")
        print("Please run the ./test-new.sh script first to generate the data.")
        return

    # 使用 pandas 读取数据
    try:
        df = pd.read_csv(csv_file)
    except pd.errors.EmptyDataError:
        print(f"Error: The CSV file '{csv_file}' is empty. No data to plot.")
        return
        
    # --- 数据预处理 ---
    # 确保所有需要的列都存在
    required_columns = ['reads_per_thread', 'step', 'stride', 'bandwidth']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: CSV file '{csv_file}' is missing one of the required columns: {required_columns}")
        return

    # 将带宽转换为数值
    df['bandwidth'] = pd.to_numeric(df['bandwidth'])

    # 获取所有独特的 step size 和 reads_per_thread 值
    step_sizes = sorted(df['step'].unique())
    rpt_values = sorted(df['reads_per_thread'].unique())
    
    print(f"Found data for step sizes: {step_sizes}")
    print(f"Found data for reads_per_thread (RPT): {rpt_values}")

    # 使用 seaborn 的色彩方案，让图更美观
    colors = sns.color_palette("viridis", n_colors=len(rpt_values))
    
    # =======================================================================
    # 核心修改点: 为每个 step 创建一张图, 图内包含多个 rpt 的曲线
    # =======================================================================
    for step in step_sizes:
        # 创建新的图表
        plt.figure(figsize=(14, 8))
        ax = plt.gca()

        # 筛选出当前 step 的数据
        step_df = df[df['step'] == step]

        # 内层循环: 遍历每个 rpt 值并绘制一条曲线
        for i, rpt in enumerate(rpt_values):
            # 从 step_df 中进一步筛选出当前 rpt 的数据
            subset = step_df[step_df['reads_per_thread'] == rpt]
            
            if not subset.empty:
                plt.plot(
                    subset['stride'], 
                    subset['bandwidth'], 
                    marker='o',          # 添加标记点
                    linestyle='-',       # 使用实线
                    markersize=5,        # 标记点大小
                    color=colors[i],     # 为不同 rpt 分配不同颜色
                    label=f'RPT = {rpt}' # 标签，用于图例
                )

        # --- 设置图表样式 (在所有曲线绘制完毕后) ---
        plt.xscale('log', base=2)
        # Y轴通常用线性刻度对比效果更好，但也可以用对数刻度
        # plt.yscale('log', base=2) 
        
        plt.title(f'Memory Bandwidth vs. Stride (Read Size = {step} Bytes)', fontsize=16, pad=20)
        plt.xlabel('Stride Interval (Bytes)', fontsize=12)
        plt.ylabel('Achieved Bandwidth (GB/s)', fontsize=12)
        
        # 设置坐标轴刻度格式
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(mticker.NullFormatter()) # 关闭次要刻度的科学计数法
        plt.xticks(step_df['stride'].unique(), rotation=45, ha="right")
        
        # 添加图例，并放置在图外，防止遮挡数据
        plt.legend(title='Reads Per Thread (RPT)', bbox_to_anchor=(1.04, 1), loc="upper left")
        
        ax.grid(True, which="both", linestyle='--', linewidth=0.5)
        
        # 调整布局以适应图例
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # 在右侧为图例留出空间
        
        # 保存图表到文件
        output_filename = f'bandwidth_plot_step_{step}_vs_rpt.png'
        plt.savefig(output_filename, bbox_inches='tight')
        print(f"Plot saved as '{output_filename}'")
        
        # 关闭当前图表以释放内存
        plt.close()

if __name__ == '__main__':
    # 调用新的绘图函数
    plot_bandwidth_data_new()