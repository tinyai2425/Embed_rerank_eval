import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def analyze_entropy_and_repeat(analysis_data, save_fig_path=None):
    """
    分析并保存可视化图像（仅保存，不再展示）

    参数:
        analysis_data: 包含 entropy 和 repeat 列的 DataFrame
        save_fig_path: 图像保存路径，必传
    """
    if not save_fig_path:
        raise ValueError("❌ 必须传入 save_fig_path，用于保存图像")

    plt.figure(figsize=(16, 12))
    
    # Entropy 图
    plt.subplot(1, 2, 1)
    entropy_data = analysis_data.entropy.sort_values()
    plt.plot(entropy_data.values, 'o', markersize=2, alpha=0.7, color='blue')
    plt.xticks(np.arange(0, len(analysis_data)+1, max(1, len(analysis_data)//10)))
    plt.xlabel("Sample Index")
    plt.ylim(9, 21)
    plt.ylabel("Entropy")
    plt.title("Entropy Values (Sorted Low to High)")
    plt.grid(True, alpha=0.3)

    # Repeat 图
    plt.subplot(1, 2, 2)
    repeat_data = analysis_data.repeat.sort_values(ascending=False)
    plt.plot(repeat_data.values, 'o', markersize=2, alpha=0.7, color='green')
    plt.xticks(np.arange(0, len(analysis_data)+1, max(1, len(analysis_data)//10)))
    plt.xlabel("Sample Index")
    plt.ylim(-0.05, 1.05)
    plt.ylabel("Repeat Value")
    plt.title("Repeat Values (Sorted High to Low)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
    plt.savefig(save_fig_path)
    print(f"[FIGURE SAVED] 图像已保存: {save_fig_path}")
    plt.close()
