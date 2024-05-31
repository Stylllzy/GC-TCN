import pandas as pd
from fancyimpute import IterativeImputer


def multiple_imputation_excel(input_filepath, output_filepath):
    # 加载数据
    df = pd.read_excel(input_filepath, sheet_name=0)

    total_missing_values = df.isna().sum().sum()
    print(f"Total missing values in DataFrame: {total_missing_values}")

    # 将数据框转换为numpy数组，因为fancyimpute接受numpy数组作为输入
    data = df.values

    # 应用MICE
    mice_imputer = IterativeImputer(verbose=2)
    data_filled = mice_imputer.fit_transform(data)

    # 将填补后的数据转换回pandas DataFrame
    # 使用原始DataFrame的列名和索引来创建一个新的DataFrame
    df_filled = pd.DataFrame(data_filled, columns=df.columns, index=df.index)

    # 保存填补后的数据到新的Excel文件
    df_filled.to_excel(output_filepath, index=False)


input_filepath = './淮安市.xlsx'  # 请替换为你的输入文件路径
output_filepath = './淮安市-填补后.xlsx'  # 指定输出文件的路径
multiple_imputation_excel(input_filepath, output_filepath)
