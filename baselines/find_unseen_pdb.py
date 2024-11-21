
import torch


data_file = "binddataset/processed/data_new.pt"

data = torch.load(data_file)
train = data.query("group == 'train' or group == 'valid'").reset_index(drop=True)
test = data.query("group == 'test' and fraction_of_this_chain > 0.8").reset_index(drop=True)
train_uid = train['uid'].tolist()
train_uid_set = set(train_uid)
test_pdb = test['pdb'].tolist()
test_uid = test['uid'].tolist()
unseen_test_pdb = []
for pdb, uid in zip(test_pdb, test_uid):
    if uid not in train_uid_set:
        unseen_test_pdb.append(pdb)

print(f"num unseen: {len(unseen_test_pdb)}")
# unseen_test_pdb_file = "binddataset/processed/unseen_test_pdb.txt"
unseen_test_pdb_file = "baselines/unseen_test_pdb.txt"
with open(unseen_test_pdb_file, 'w') as f:
    f.write(' '.join(unseen_test_pdb))




