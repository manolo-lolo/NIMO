import torch
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelSummary, ModelCheckpoint

from hybrid_lasso.model_simple_nn import SimpleNN
from hybrid_lasso.dataset_synthetic import SyntheticDataset

import argparse
import numpy as np


def get_dataloaders(data_name, data_path, batch_size):
    if data_name == "synthetic":
        train_dataset = SyntheticDataset(data_path, split="train")
        val_dataset = SyntheticDataset(data_path, split="val")
        test_dataset = SyntheticDataset(data_path, split="test")
    else:
        raise NotImplementedError

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader


def main(args):
    if args.expr_name is not None:
        expr_name = args.expr_name
    else:
        expr_name = f"{args.data_name}"
    logger = TensorBoardLogger("tb_logs", name=expr_name)

    # fix seed
    torch.manual_seed(42)
    np.random.seed(42)

    # get dataset and dataloader
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_name, args.data_path, args.batch_size
    )

    model = SimpleNN(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        task=args.task,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )

    if args.task == "reg":
        callbacks = [
            ModelCheckpoint(
                monitor="val_loss", mode="min", save_top_k=1, filename="best_model_{epoch}"
            ),
        ]
        if args.early_stop:
            callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=10))
    elif args.task == "cls":
        callbacks = [
            ModelCheckpoint(
                monitor="val_acc", mode="max", save_top_k=1, filename="best_model_{epoch}"
            ),
        ]
        if args.early_stop:
            callbacks.append(EarlyStopping(monitor="val_acc", mode="max", patience=10))

    # trainer = L.Trainer(
    #     max_epochs=args.num_epochs,
    #     callbacks=callbacks,
    #     logger=logger,
    #     log_every_n_steps=50,
    #     check_val_every_n_epoch=100
    # )

    trainer = L.Trainer(
        max_epochs=args.num_epochs,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
    )

    trainer.fit(model, train_loader, val_loader)

    # ========================= Predict ========================
    model = SimpleNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    predictions = trainer.predict(model, test_loader)
    torch.save(predictions, f"./ckpts/predictions_{expr_name}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data setting
    parser.add_argument("--data_path", type=str, default="./data/boston_1")
    parser.add_argument("--data_name", type=str, default="synthetic", help="synthetic")
    parser.add_argument("--batch_size", type=int, default=1024)

    # model setting
    parser.add_argument("--input_dim", type=int, default=15)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--task", type=str, default="reg", choices=["reg", "cls"])
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0)

    # training setting
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--expr_name", type=str, default=None)
    parser.add_argument("--early_stop", action="store_true", help="whether to use early stopping")

    args = parser.parse_args()
    main(args)
