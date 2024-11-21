
import pandas as pd
import numpy as np


def analyze(df, metric, method="dynamicbind"):
    pdb_list = df['entryName'].tolist()
    pdb_unique = list(set(pdb_list))
    grouped_metrics = df.groupby("entryName")[metric].apply(list)
    metrics_list = [x[0] for x in grouped_metrics.tolist()]
    metrics_list = np.array(metrics_list)
    metrics_mean = np.mean(metrics_list)
    metrics_2A = (metrics_list < 2).sum() / metrics_list.shape[0]
    metrics_5A = (metrics_list < 5).sum() / metrics_list.shape[0]

    metrics_25 = np.percentile(metrics_list, 25)
    metrics_50 = np.percentile(metrics_list, 50)
    metrics_75 = np.percentile(metrics_list, 75)
    print(f"-----------------------------{method}-----------------------------")
    print(f"num samples: {len(pdb_unique)}")
    print(f"{metric}: {metrics_mean}, {metric} < 2A: {metrics_2A}, {metric} < 5A: {metrics_5A}, "
          f"{metric} 25%: {metrics_25}, {metric} 50%: {metrics_50}, {metric} 75%: {metrics_75}")


def analyze_unseen(df, metric, seleted_pdb, method="dynamicbind"):
    filtered_df = df[df['entryName'].isin(seleted_pdb)]
    pdb_list = filtered_df['entryName'].tolist()
    pdb_unique = list(set(pdb_list))
    grouped_metrics = filtered_df.groupby("entryName")[metric].apply(list)
    metrics_list = [x[0] for x in grouped_metrics.tolist()]
    metrics_list = np.array(metrics_list)
    metrics_mean = np.mean(metrics_list)
    metrics_2A = (metrics_list < 2).sum() / metrics_list.shape[0]
    metrics_5A = (metrics_list < 5).sum() / metrics_list.shape[0]

    metrics_25 = np.percentile(metrics_list, 25)
    metrics_50 = np.percentile(metrics_list, 50)
    metrics_75 = np.percentile(metrics_list, 75)
    print(f"-----------------------------{method}-----------------------------")
    print(f"num samples: {len(pdb_unique)}")
    print(f"{metric}: {metrics_mean}, {metric} < 2A: {metrics_2A}, {metric} < 5A: {metrics_5A}, "
          f"{metric} 25%: {metrics_25}, {metric} 50%: {metrics_50}, {metric} 75%: {metrics_75}")


def analyze_fabind(res_file, seleted_pdb, method="fabind+"):
    pdb_list = []
    rmsd_list = []
    with open(res_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        pdb, rmsd = line[0], float(line[1])
        if (seleted_pdb is not None) and (pdb not in seleted_pdb):
            continue
        pdb_list.append(pdb)
        rmsd_list.append(rmsd)
    rmsd_list = np.array(rmsd_list)
    rmsd_mean = np.mean(rmsd_list)
    rmsd_2A = (rmsd_list < 2).sum() / rmsd_list.shape[0]
    rmsd_5A = (rmsd_list < 5).sum() / rmsd_list.shape[0]

    rmsd_25 = np.percentile(rmsd_list, 25)
    rmsd_50 = np.percentile(rmsd_list, 50)
    rmsd_75 = np.percentile(rmsd_list, 75)
    print(f"-----------------------------{method}-----------------------------")
    print(f"num samples: {len(pdb_list)}")
    print(f"rmsd: {rmsd_mean}, rmsd < 2A: {rmsd_2A}, rmsd < 5A: {rmsd_5A}, "
          f"rmsd 25%: {rmsd_25}, rmsd 50%: {rmsd_50}, rmsd 75%: {rmsd_75}")



if __name__ == '__main__':
    file_name = "./all_baselines.xlsx"
    df = pd.read_excel(file_name, sheet_name="Fig2")
    # df_Vina = df.query("method == 'Vina' and dataset == 'PDBBind'")
    # analyze(df_Vina, "rmsd", method="Vina")
    # df_Glide = df.query("method == 'Glide' and dataset == 'PDBBind'")
    # analyze(df_Glide, "rmsd", method="Glide")
    # df_Gnina = df.query("method == 'Gnina' and dataset == 'PDBBind'")
    # analyze(df_Gnina, "rmsd", method="Gnina")
    # df_Tankbind = df.query("method == 'TankBind' and dataset == 'PDBBind'")
    # analyze(df_Tankbind, "rmsd", method="TankBind")
    # df_DiffDock = df.query("method == 'DiffDock' and dataset == 'PDBBind'")
    # analyze(df_DiffDock, "rmsd", method="DiffDock")
    # df_dynamic = df.query("method == 'DynamicBind' and dataset == 'PDBBind'")
    # analyze(df_dynamic, "rmsd", method="DynamicBind")
    # # analyze(df_dynamic, "pocket_rmsd", method="DynamicBind")

    fabind_file = './fabind_rmsd.txt'
    fabind_plus_file = "./fabind_plus_rmsd.txt"
    R2Fdock_file = "./R2FDock_rmsd.txt"
    analyze_fabind(fabind_file, None, method='fabind')
    analyze_fabind(fabind_plus_file, None, method='fabind+')
    analyze_fabind(R2Fdock_file, None, method='R2FDock')

    print(f"----------------------------------------------------------Unseen protein----------------------------------------------------------")
    unseen_file = "./unseen_test_pdb.txt"
    with open(unseen_file, 'r') as f:
        unseen_pdbs = f.read()
    unseen_pdbs = unseen_pdbs.strip('\n').split(' ')
    # analyze_unseen(df_Vina, "rmsd", unseen_pdbs, method="Vina")
    # analyze_unseen(df_Glide, "rmsd", unseen_pdbs, method="Glide")
    # analyze_unseen(df_Gnina, "rmsd", unseen_pdbs, method="Gnina")
    # analyze_unseen(df_Tankbind, "rmsd", unseen_pdbs, method="TankBind")
    # analyze_unseen(df_DiffDock, "rmsd", unseen_pdbs, method="DiffDock")
    # analyze_unseen(df_dynamic, "rmsd", unseen_pdbs, method="DynamicBind")
    analyze_fabind(fabind_file, unseen_pdbs, method='fabind')
    analyze_fabind(fabind_plus_file, unseen_pdbs, method='fabind+')
    analyze_fabind(R2Fdock_file, unseen_pdbs, method='R2FDock')