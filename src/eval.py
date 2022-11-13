import torch
import argparse
from pathlib import Path
from tqdm import tqdm

from wayformer import WayformerPL
from utils import Args, batch_list_to_batch_tensors
from dataset_argoverse import Dataset
from torch.utils.data import DataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str)
    user_args = parser.parse_args()
    res_path = Path(user_args.ckpt).parent.parent / 'eval.txt'
    print(res_path)
    model: WayformerPL = WayformerPL.load_from_checkpoint(
        checkpoint_path=user_args.ckpt
    )
    model.to('cuda:0')
    model.eval()

    args = model.args
    val_data = Dataset(args, 'val')
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.data_workers//2,
        collate_fn=batch_list_to_batch_tensors
    )
    num_samples = 0
    ADEs = []
    FDEs = []
    missed = 0

    for batch in tqdm(val_loader):
        metrics = model.evaluate_model(batch, k=6)
        ADEs.append(metrics.minADE)
        FDEs.append(metrics.minFDE)
        missed += metrics.missed
        num_samples += len(batch)

    missRate = missed / num_samples
    ADE = sum(ADEs) / len(ADEs)
    FDE = sum(FDEs) / len(FDEs)
    with open(res_path, 'w') as f:
        f.write(f'minADE: {ADE}\nminFDE: {FDE}\nmissRate: {missRate}\n')