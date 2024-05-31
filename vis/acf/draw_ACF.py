import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def process_and_plot(file_path, city_name, output_folder='./'):
    # 读取数据
    data = pd.read_csv(file_path, encoding='ISO-8859-1')

    # 清理 'close' 列中的 NaN 值
    close_data = data['close'].dropna()

    plt.rcParams.update({
        'font.size': 20,  # 修改字体大小
        'font.family': 'Times New Roman',
        'xtick.major.size': 5,  # x轴刻度大小
        'xtick.major.width': 1,  # x轴刻度宽度
        'ytick.major.size': 5,  # y轴刻度大小
        'ytick.major.width': 1,  # y轴刻度宽度
        'axes.linewidth': 1.5,  # 轴线宽度
    })

    # 创建ACF和PACF图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 绘制ACF图
    plot_acf(close_data, ax=ax1, lags=40, alpha=0.05)
    ax1.set_title(f'ACF - {city_name}', fontsize=22)
    ax1.grid(False)  # 去除背景网格

    # 绘制PACF图
    plot_pacf(close_data, ax=ax2, lags=40, alpha=0.05, method='ywm')
    ax2.set_title(f'PACF - {city_name}', fontsize=22)
    ax2.grid(False)  # 去除背景网格

    plt.tight_layout()

    # 保存为高分辨率图片
    high_res_image_path = output_folder + f'acf_pacf_{city_name.lower()}.pdf'
    fig.savefig(high_res_image_path, dpi=600, bbox_inches='tight')
    plt.show()

    return high_res_image_path


# 处理上海的数据
sh_image_path = process_and_plot('../../dataset/tanjiaoyi/sh_carbon.csv', 'Shanghai')

# 处理广东的数据
# gd_image_path = process_and_plot('../../dataset/tanjiaoyi/gd_carbon.csv', 'Guangdong')

# 处理湖北的数据
# hb_image_path = process_and_plot('../../dataset/tanjiaoyi/hb_carbon.csv', 'Hubei')

print(f"上海数据图表已保存至: {sh_image_path}")
# print(f"广东数据图表已保存至: {gd_image_path}")
# print(f"湖北数据图表已保存至: {hb_image_path}")
