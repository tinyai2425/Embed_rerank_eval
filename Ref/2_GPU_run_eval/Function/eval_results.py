import pandas as pd

def evaluate_reference_results(
    result_get,
    verbal=False,
    save_summary_path=None,
    save_writer=None,
    sheet_prefix=""
):
    df_results = result_get
    summaries = []
    grouped_by_flavor = df_results.groupby(["project_name", "flavor"])
    for (project_name, flavor), df_flavor in grouped_by_flavor:
        summaries.append({
            "Project": project_name,
            "Flavor": flavor,
            "Test No.": len(df_flavor),
            "prompt len": round(df_flavor['prompt_token_len'].mean(), 2),
            "response len": round(df_flavor['response_token_len'].mean(), 2),
            "get_ans": round(df_flavor['get_ans'].mean(), 2),
            "rp>0.1": round((df_flavor['repeat'] > 0.1).mean(), 4),
            "rp@99%": round(df_flavor['repeat'].quantile(0.99), 5),
            "ent<14.5": round((df_flavor['entropy'] < 14.5).mean(), 4),
            "ent@1%": round(df_flavor['entropy'].quantile(0.01), 2),
            "TTFT": round(df_flavor['first_token_time'].mean(), 2),
            "TPS": round(df_flavor['response_token_len'].sum() / df_flavor['decode_time'].sum(), 2),
            "ACC": round(df_flavor['correct'].mean() * 100, 2)
        })

    df_summary = pd.DataFrame(summaries)
    print(f"\n[Summary] {sheet_prefix}")
    print(df_summary.to_string(index=False))

    df_breakdown = None
    if verbal:
        print(f"\n[Breakdown] {sheet_prefix}")
        details = []
        grouped_by_vertical = df_results.groupby(["project_name", "flavor", "vertical"])
        for (project_name, flavor, vertical), df_vertical in grouped_by_vertical:
            details.append({
                "Project": project_name,
                "Flavor": flavor,
                "Vertical": vertical,
                "样本数": len(df_vertical),
                "prompt len": round(df_vertical["prompt_token_len"].mean(), 2),
                "response_token_len": round(df_vertical["response_token_len"].mean(), 2),
                "TTFT": round(df_vertical["first_token_time"].mean(), 2),
                "TPS": round(
                    df_vertical["response_token_len"].sum() / max(1e-5, (df_vertical["total_time"].sum() - df_vertical["first_token_time"].sum())),
                    2),
                "ACC": round(df_vertical["correct"].mean() * 100, 2)
            })

        df_breakdown = pd.DataFrame(details)
        print(df_breakdown.to_string(index=False))

    # === 自动写入 Excel（如果提供保存路径）
    if save_summary_path:
        with pd.ExcelWriter(save_summary_path) as writer:
            # === 修正 sheet 名拼接逻辑 ===
            if sheet_prefix:
                summary_sheet = f"summary_{sheet_prefix}"[:31]
                breakdown_sheet = f"breakdown_{sheet_prefix}"[:31]
            else:
                summary_sheet = "summary"
                breakdown_sheet = "breakdown"

            df_summary.to_excel(writer, sheet_name=summary_sheet, index=False)
            print(f"[Excel] Summary saved to sheet: {summary_sheet}")

            if verbal and df_breakdown is not None:
                df_breakdown.to_excel(writer, sheet_name=breakdown_sheet, index=False)
                print(f"[Excel] Breakdown saved to sheet: {breakdown_sheet}")

    # 自动打开 writer（如果路径传入了，但 writer 没传）
    internal_writer = None
    if save_summary_path and save_writer is None:
        internal_writer = pd.ExcelWriter(save_summary_path)
        save_writer = internal_writer

    # 写入 Excel
    if save_writer:
        summary_sheet = f"summary_{sheet_prefix}" if sheet_prefix else "summary"
        breakdown_sheet = f"breakdown_{sheet_prefix}" if sheet_prefix else "breakdown"

        df_summary.to_excel(save_writer, sheet_name=summary_sheet, index=False)
        print(f"[Excel] Summary written to: {summary_sheet}")

        if verbal and df_breakdown is not None:
            df_breakdown.to_excel(save_writer, sheet_name=breakdown_sheet, index=False)
            print(f"[Excel] Breakdown written to: {breakdown_sheet}")

    if internal_writer:
        save_writer.close()


    return df_results
