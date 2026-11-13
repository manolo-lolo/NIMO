import os
import torch
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from hybrid_lasso.model_ar_reg import AdaptiveRidgeRegression
from hybrid_lasso.model_ar_logistic import AdaptiveRidgeLogisticRegression

from hybrid_lasso.dataset_synthetic import SyntheticDataset

import argparse
import numpy as np


model_classes = {
    "reg": AdaptiveRidgeRegression,
    "cls": AdaptiveRidgeLogisticRegression,
}


def get_dataloaders(data_name, data_path, batch_size):
    if data_name == "synthetic":
        train_dataset = SyntheticDataset(data_path, split="train")
        val_dataset = SyntheticDataset(data_path, split="val")
        test_dataset = SyntheticDataset(data_path, split="test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader


def main(args):
    if args.expr_name is not None:
        expr_name = args.expr_name
    else:
        expr_name = f"{args.data_name}_adaptive_ridge"
    logger = TensorBoardLogger("tb_logs", name=expr_name)

    # fix seed
    torch.manual_seed(42)
    np.random.seed(42)

    # get dataset and dataloader
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_name, args.data_path, args.batch_size
    )

    if args.task == "reg":
        model = AdaptiveRidgeRegression(
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            learning_rate=args.learning_rate,
            lasso_penalty=args.lasso_penalty,
            group_penalty=args.group_penalty,
            lasso_norm=args.lasso_norm,
            group_norm=args.group_norm,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            fit_intercept=args.fit_intercept,
            noise_injection=args.noise_injection,
            no_feat_mask=args.no_feat_mask,
            no_pos_encoding=args.no_pos_encoding,
        )
    elif args.task == "cls":
        model = AdaptiveRidgeLogisticRegression(
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            learning_rate=args.learning_rate,
            lasso_penalty=args.lasso_penalty,
            group_penalty=args.group_penalty,
            lasso_norm=args.lasso_norm,
            group_norm=args.group_norm,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            fit_intercept=args.fit_intercept,
        )
    else:
        raise ValueError("Invalid task")

    if args.task == "reg":
        callbacks = [
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                filename="best_model_{epoch}",
            ),
        ]
        if args.early_stop:
            callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=10))
    elif args.task == "cls":
        callbacks = [
            ModelCheckpoint(
                monitor="val_acc",
                mode="max",
                save_top_k=1,
                filename="best_model_{epoch}",
            ),
        ]
        if args.early_stop:
            callbacks.append(EarlyStopping(monitor="val_acc", mode="max", patience=1000))

    # trainer = L.Trainer(
    #     max_epochs=args.num_epochs,
    #     logger=logger,
    #     log_every_n_steps=50,
    #     callbacks=callbacks,
    #     check_val_every_n_epoch=100
    # )

    trainer = L.Trainer(
        max_epochs=args.num_epochs,
        logger=logger,
        log_every_n_steps=1,
        callbacks=callbacks,
        check_val_every_n_epoch=1,
    )

    try:
        trainer.fit(model, train_loader, val_loader)
        print("==> Beta at the end of training:", model.beta.cpu().detach().numpy())
        print("==> Beta_0 at the end of training:", model.beta_0.cpu().detach().numpy())

        # load the best model
        model = model_classes[args.task].load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )
        print("==> Beta of the best model:", model.beta.cpu().detach().numpy())
        print("==> Beta_0 of the best model:", model.beta_0.cpu().detach().numpy())

        # check and create the ckpts directory
        if not os.path.exists("./ckpts"):
            os.makedirs("./ckpts")

        # save beta
        beta = [model.beta, model.beta_0]
        torch.save(beta, f"./ckpts/beta_{expr_name}.pt")

        # get predictions
        predictions = trainer.predict(model, test_loader)
        torch.save(predictions, f"./ckpts/predictions_{expr_name}.pt")

        # get nonlinear effects
        if args.task == "reg":
            model.to("cuda")
            nn_effects = []
            for batch in test_loader:
                X = batch["features"]
                nn_out = model.compute_nonlinearity(X.to("cuda"))
                nn_out = nn_out.cpu().detach().numpy()
                nn_effects.append(nn_out)
            nn_effects = np.concatenate(nn_effects, axis=0)
            np.save(f"./ckpts/nn_effects_{expr_name}.npy", nn_effects)
            print(f"==> Nonlinear effects saved at ./ckpts/nn_effects_{expr_name}.npy")

    except KeyboardInterrupt:
        print("\n==> Training interrupted. Saving latest checkpoint and computing predictions...")

        # Save the current state before exiting
        ckpt_path = f"./ckpts/interrupt_{expr_name}.ckpt"
        trainer.save_checkpoint(ckpt_path)
        print(f"Checkpoint saved at {ckpt_path}")

        # Compute predictions using the current state
        model.eval()  # Ensure the model is in evaluation mode
        predictions = trainer.predict(model, test_loader)
        torch.save(predictions, f"./ckpts/interrupted_predictions_{expr_name}.pt")
        print(f"Predictions saved at ./ckpts/interrupted_predictions_{expr_name}.pt")

        # Save the last beta values
        beta = [model.beta, model.beta_0]
        torch.save(beta, f"./ckpts/interrupted_beta_{expr_name}.pt")
        print(f"Beta values saved at ./ckpts/interrupted_beta_{expr_name}.pt")

        print("Exiting gracefully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default="reg", choices=["reg", "cls", "reg_add"])

    parser.add_argument("--data_name", type=str, default="synthetic")
    parser.add_argument("--data_path", type=str, default="./data/synthetic_reg/case1")
    parser.add_argument("--batch_size", type=int, default=1024)

    parser.add_argument("--input_dim", type=int, default=3)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--lasso_penalty", type=float, default=1e-3)
    parser.add_argument("--group_penalty", type=float, default=1e-3)
    parser.add_argument("--lasso_norm", type=float, default=1)
    parser.add_argument("--group_norm", type=float, default=1)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0)

    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--expr_name", type=str, default=None)
    parser.add_argument("--fit_intercept", action="store_true", default=False)
    parser.add_argument("--noise_injection", action="store_true", default=False)
    parser.add_argument("--no_feat_mask", action="store_true", default=False)
    parser.add_argument("--no_pos_encoding", action="store_true", default=False)

    args = parser.parse_args()
    main(args)
