import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from utils import load_config, Args, batch_list_to_batch_tensors
from wayformer import WayformerPL
from dataset_argoverse import Dataset


def train_model(
    model: WayformerPL
):
    args = model.args
    logger = TensorBoardLogger(save_dir=f'{args.output_dir}', name=args.exp_name)
    callbacks = [
        LearningRateMonitor(),
        ModelCheckpoint(
            filename='{epoch}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            save_last=True
        ),
    ]
    trainer = pl.Trainer(
        logger=logger,
        log_every_n_steps=args.log_period,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        accelerator='gpu',
        devices=args.num_gpu,
        strategy=DDPStrategy(find_unused_parameters=False)
    )
    train_set = Dataset(args, 'train')
    val_set = Dataset(args, 'val')
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.data_workers,
        collate_fn=batch_list_to_batch_tensors
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.data_workers,
        collate_fn=batch_list_to_batch_tensors
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

# def eval_model(
#     model: pl.LightningModule,
#     args: Args
# ):
#     pass


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config_dir', type=str, default='../configs/wayformer.1.json')
    args_run = parser.parse_args()
    args: Args = load_config(args_run.config_dir)
    pl.seed_everything(0)
    model = WayformerPL(args)
    train_model(model)