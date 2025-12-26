"""Training function using Lightning Trainer."""

import os
from pathlib import Path

import gymnasium as gym
import lightning as L
import mlflow
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from crypto_dqn_ops.data.data_loader import prepare_data_for_training
from crypto_dqn_ops.training.lightning_trainer import RainbowLightningModule
from crypto_dqn_ops.training.rl_datamodule import RLDataModule
from crypto_dqn_ops.utils.helpers import seed_everything


def train_with_lightning(cfg):
    """Train Rainbow DQN using Lightning Trainer.

    Args:
        cfg: Hydra configuration object
    """
    print("=" * 80)
    print("Training with Lightning Trainer")
    print("=" * 80)

    seed_everything(cfg.seed)

    data_path = Path(cfg.data.data_path)
    useful_data, train_data, test_data = prepare_data_for_training(
        data_path,
        name_num=cfg.data.crypto_index,
        train_split=cfg.data.train_split,
        use_dvc=cfg.dvc.enabled,
    )

    gym.envs.register(
        id=cfg.data.env_id,
        entry_point="crypto_dqn_ops.environment.crypto_env:CryptoEnv",
    )

    env = gym.make(
        cfg.data.env_id,
        data=train_data,
        wnd_t=cfg.data.window_size,
        cycle_T=cfg.data.cycle_length,
    )

    obs_dim = env.observation_space.shape[1]
    action_dim = env.action_space.n

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    model_dir = Path(cfg.model_dir) / f"{cfg.data.name}_{cfg.seed}"
    model_dir.mkdir(parents=True, exist_ok=True)

    git_commit = os.popen("git rev-parse HEAD").read().strip()

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
        run_name=f"{cfg.data.name}_lightning_training",
    )

    mlflow_logger.log_hyperparams(
        {
            "seed": cfg.seed,
            "crypto": cfg.data.name,
            "window_size": cfg.data.window_size,
            "cycle_length": cfg.data.cycle_length,
            "num_frames": cfg.training.num_frames,
            "batch_size": cfg.training.batch_size,
            "learning_rate": cfg.training.learning_rate,
            "gamma": cfg.training.gamma,
            "memory_size": cfg.training.memory_size,
            "n_step": cfg.training.n_step,
            "v_min": cfg.model.v_min,
            "v_max": cfg.model.v_max,
            "atom_size": cfg.model.atom_size,
            "git_commit": git_commit,
            "framework": "Lightning",
        }
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename="model_{epoch:05d}",
        every_n_train_steps=cfg.training.checkpoint_interval,
        save_top_k=-1,
    )

    lightning_module = RainbowLightningModule(
        obs_dim=obs_dim,
        action_dim=action_dim,
        memory_size=cfg.training.memory_size,
        batch_size=cfg.training.batch_size,
        target_update=cfg.training.target_update,
        gamma=cfg.training.gamma,
        lr=cfg.training.learning_rate,
        alpha=cfg.training.per_alpha,
        beta=cfg.training.per_beta,
        prior_eps=cfg.training.prior_eps,
        v_min=cfg.model.v_min,
        v_max=cfg.model.v_max,
        atom_size=cfg.model.atom_size,
        n_step=cfg.training.n_step,
    )

    datamodule = RLDataModule(
        env=env,
        agent=lightning_module,
        num_frames=cfg.training.num_frames,
        batch_size=cfg.training.batch_size,
    )

    trainer = L.Trainer(
        max_epochs=-1,
        max_steps=cfg.training.num_frames,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback],
        accelerator=cfg.training.get("accelerator", "cpu"),
        devices=cfg.training.get("devices", 1),
        precision=cfg.training.get("precision", 32),
        log_every_n_steps=cfg.training.get("log_interval", 50),
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    print("Starting Lightning training...")
    trainer.fit(lightning_module, datamodule=datamodule)

    final_model_path = model_dir / "model_final.pth"
    torch.save(lightning_module.dqn.state_dict(), final_model_path)
    print("Training completed! Final model saved to {}".format(final_model_path))

    env.close()
