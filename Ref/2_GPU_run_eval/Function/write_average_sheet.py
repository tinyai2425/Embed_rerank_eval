import pandas as pd
from tabulate import tabulate
import os

def save_overall_summary(save_summary_path, prefix, type):
    xls = pd.ExcelFile(save_summary_path)
    target_sheets = [s for s in xls.sheet_names if s.startswith(f"{prefix}")]

    if not target_sheets:
        print(f"No sheet starts with '{prefix}' found in {save_summary_path}")
        return

    print(f"Aggregating sheets: {target_sheets}")

    # 合并数据
    df_list = [pd.read_excel(xls, sheet_name=s) for s in target_sheets]
    combined = pd.concat(df_list, ignore_index=True)

    # 主键和数值列
    group_keys = [col for col in ["Project", "Flavor", "Vertical"] if col in combined.columns]
    numeric_cols = combined.select_dtypes(include='number').columns.difference(group_keys).tolist()

    # 均值
    df_avg = combined.groupby(group_keys)[numeric_cols].mean().reset_index()

    # 提取非数值列（如 Test No.）用于还原完整表头
    df_first = pd.read_excel(xls, sheet_name=target_sheets[0])
    for col in combined.columns:
        if col not in numeric_cols + group_keys and col in df_first.columns:
            values = df_first[[*group_keys, col]]
            df_avg = pd.merge(df_avg, values, on=group_keys, how="left")

    # 排序列顺
    ordered_cols = [col for col in df_first.columns if col in df_avg.columns]
    df_avg = df_avg[ordered_cols]

    def format_float(x):
        if isinstance(x, float):
            x = round(x, 4)
            return str(x).rstrip('0').rstrip('.') if '.' in str(x) else str(x)
        return x

    df_avg_formatted = df_avg.applymap(format_float)

    md_path = os.path.join(os.path.dirname(save_summary_path), f"{prefix}_{type}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(tabulate(df_avg_formatted, headers="keys", tablefmt="github"))

    print(f"Markdown summary saved: {md_path}")