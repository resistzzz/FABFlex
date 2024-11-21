
import pandas as pd
import numpy as np


def get_lddt_clash(df, metric, method):
    df_method = df.query(f"method == '{method}' and dataset == 'PDBBind'")
    pdb_list = df_method['entryName'].tolist()
    pdb_unique = list(set(pdb_list))
    grouped_metrics = df_method.groupby("entryName")[metric].apply(list)
    metrics_list = [x[0] for x in grouped_metrics.tolist()]
    metrics_list = np.array(metrics_list)
    metrics_mean = np.mean(metrics_list)

    print(f"-----------------------------{method}: {metric}-----------------------------")
    print(f"num samples: {len(pdb_unique)}")
    print(f"Mean {metric}: {metrics_mean}")


if __name__ == '__main__':
    file_name = "./all_baselines.xlsx"
    df = pd.read_excel(file_name, sheet_name="Fig2")
    get_lddt_clash(df, "clashScore", "Vina")
    get_lddt_clash(df, "clashScore", "Glide")
    get_lddt_clash(df, "clashScore", "Gnina")
    get_lddt_clash(df, "clashScore", "TankBind")
    get_lddt_clash(df, "clashScore", "DiffDock")
    get_lddt_clash(df, "clashScore", "DynamicBind")

    get_lddt_clash(df, "predicted_lddt", "Vina")
    get_lddt_clash(df, "predicted_lddt", "Glide")
    get_lddt_clash(df, "predicted_lddt", "Gnina")
    get_lddt_clash(df, "predicted_lddt", "TankBind")
    get_lddt_clash(df, "predicted_lddt", "DiffDock")
    get_lddt_clash(df, "predicted_lddt", "DynamicBind")
