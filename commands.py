"""Main command-line interface for training and inference."""

import os
from pathlib import Path

import fire
import gymnasium as gym
import matplotlib.pyplot as plt
import mlflow
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

from crypto_dqn_ops.agents.rainbow_agent import RainbowAgent
from crypto_dqn_ops.data.data_loader import prepare_data_for_training
from crypto_dqn_ops.inference.predictor import CryptoPredictor
from crypto_dqn_ops.training.train_with_lightning import train_with_lightning
from crypto_dqn_ops.utils.helpers import seed_everything


def train(config_name: str = "config", **kwargs):
    """Train Rainbow DQN model.

    Args:
        config_name: Name of config file (without .yaml)
        **kwargs: Config overrides (e.g., data='eth', training__batch_size=256)
                  Use double underscore __ for nested configs
    """
    overrides = []
    for key, value in kwargs.items():
        key_path = key.replace("__", ".")
        overrides.append("{}={}".format(key_path, value))

    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name=config_name, overrides=overrides)

    print("=" * 80)
    print("Training Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    if cfg.training.get("use_lightning", False):
        print("Using Lightning Trainer for training...")
        train_with_lightning(cfg)
        return

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
        cfg.data.env_id, data=train_data, wnd_t=cfg.data.window_size, cycle_T=cfg.data.cycle_length
    )

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name=f"{cfg.data.name}_training"):
        mlflow.log_params(
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
                "git_commit": os.popen("git rev-parse HEAD").read().strip(),
            }
        )

        agent = RainbowAgent(
            env=env,
            memory_size=cfg.training.memory_size,
            batch_size=cfg.training.batch_size,
            target_update=cfg.training.target_update,
            gamma=cfg.training.gamma,
            alpha=cfg.training.per_alpha,
            beta=cfg.training.per_beta,
            prior_eps=cfg.training.prior_eps,
            v_min=cfg.model.v_min,
            v_max=cfg.model.v_max,
            atom_size=cfg.model.atom_size,
            n_step=cfg.training.n_step,
            lr=cfg.training.learning_rate,
        )

        state, _ = env.reset()
        update_cnt = 0
        losses = []
        scores = []
        mean_scores = []
        score = 0

        model_dir = Path(cfg.model_dir) / f"{cfg.data.name}_{cfg.seed}"
        model_dir.mkdir(parents=True, exist_ok=True)

        for frame_idx in range(1, cfg.training.num_frames + 1):
            action = agent.select_action(state)
            next_state, reward, done = agent.step(action)

            state = next_state
            score += reward

            fraction = min(frame_idx / cfg.training.num_frames, 1.0)
            agent.beta = agent.beta + fraction * (1.0 - agent.beta)

            if done:
                state, _ = env.reset()
                scores.append(score)
                if len(scores) >= 10:
                    mean_score = sum(scores[-10:]) / 10
                    mean_scores.append(mean_score)
                    mlflow.log_metric("mean_score", mean_score, step=frame_idx)
                score = 0

            if len(agent.memory) >= agent.batch_size:
                loss = agent.update_model()
                losses.append(loss)
                update_cnt += 1

                if update_cnt % agent.target_update == 0:
                    agent._target_hard_update()

                if frame_idx % cfg.training.log_interval == 0:
                    mlflow.log_metric("loss", loss, step=frame_idx)
                    mlflow.log_metric("beta", agent.beta, step=frame_idx)

            if frame_idx % cfg.training.checkpoint_interval == 0:
                checkpoint_path = model_dir / "model_{}.pth".format(frame_idx)
                torch.save(agent.dqn.state_dict(), checkpoint_path)
                print(f"Checkpoint saved at frame {frame_idx}: {checkpoint_path}")

            if frame_idx % 10000 == 0:
                avg_loss = sum(losses[-1000:]) / len(losses[-1000:]) if losses else 0
                avg_score = sum(mean_scores[-10:]) / len(mean_scores[-10:]) if mean_scores else 0
                print(
                    "Frame: {}/{}, Avg Loss: {:.4f}, Avg Score: {:.4f}".format(
                        frame_idx, cfg.training.num_frames, avg_loss, avg_score
                    )
                )

        final_model_path = model_dir / "model_final.pth"
        torch.save(agent.dqn.state_dict(), final_model_path)
        mlflow.log_artifact(str(final_model_path))

        plots_dir = Path(cfg.plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)

        if losses:
            plt.figure(figsize=(10, 6))
            plt.plot(losses)
            plt.title("Training Loss")
            plt.xlabel("Update Step")
            plt.ylabel("Loss")
            plt.grid(True)
            loss_plot_path = plots_dir / "loss.png"
            plt.savefig(loss_plot_path)
            plt.close()
            mlflow.log_artifact(str(loss_plot_path))

        if mean_scores:
            plt.figure(figsize=(10, 6))
            plt.plot(mean_scores)
            plt.title("Mean Score (Last 10 Episodes)")
            plt.xlabel("Episode")
            plt.ylabel("Mean Score")
            plt.grid(True)
            score_plot_path = plots_dir / "mean_score.png"
            plt.savefig(score_plot_path)
            plt.close()
            mlflow.log_artifact(str(score_plot_path))

        print("Training completed! Final model saved to {}".format(final_model_path))
        print("Plots saved to {}".format(plots_dir))

    env.close()


def infer(model_path: str, data_path: str = "data/crypto_data.pkl", crypto: str = "BTC"):
    """Run inference with trained model.

    Args:
        model_path: Path to trained model
        data_path: Path to data file
        crypto: Cryptocurrency name (BTC or ETH)
    """
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="config")

    crypto_index = 0 if crypto.upper() == "BTC" else 1

    useful_data, train_data, test_data = prepare_data_for_training(
        Path(data_path), name_num=crypto_index, train_split=cfg.data.train_split, use_dvc=False
    )

    obs_dim = cfg.data.window_size + 2

    predictor = CryptoPredictor(
        model_path=model_path,
        obs_dim=obs_dim,
        action_dim=2,
        v_min=cfg.model.v_min,
        v_max=cfg.model.v_max,
        atom_size=cfg.model.atom_size,
    )

    print(f"Running inference on {crypto} test data...")
    print(f"Test data length: {len(test_data)} days")

    gym.envs.register(
        id=cfg.data.env_id,
        entry_point="crypto_dqn_ops.environment.crypto_env:CryptoEnv",
    )

    test_env = gym.make(
        cfg.data.env_id, data=test_data, wnd_t=cfg.data.window_size, cycle_T=cfg.data.cycle_length
    )

    state, _ = test_env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < 1000:
        action = predictor.predict(state)
        state, reward, done, _ = test_env.step(action)
        total_reward += reward
        steps += 1

    print("Inference completed!")
    print("Total steps: {}".format(steps))
    print("Total reward: {:.4f}".format(total_reward))

    test_env.close()


def export_onnx(model_path: str, output_path: str = "trained_models/model.onnx"):
    """Export model to ONNX format.

    Args:
        model_path: Path to trained PyTorch model
        output_path: Path to save ONNX model
    """
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="config")

    from crypto_dqn_ops.models.rainbow_network import Network

    obs_dim = cfg.data.window_size + 2
    action_dim = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    support = torch.linspace(cfg.model.v_min, cfg.model.v_max, cfg.model.atom_size).to(device)

    model = Network(obs_dim, action_dim, cfg.model.atom_size, support).to(device)

    if Path(model_path).exists():
        state_dict = torch.load(model_path, map_location=device)
        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict)
        else:
            model = state_dict
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")

    model.eval()

    dummy_input = torch.randn(1, obs_dim).to(device)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["q_values"],
        dynamic_axes={"observation": {0: "batch_size"}, "q_values": {0: "batch_size"}},
    )

    print(f"Model exported to ONNX: {output_path}")


def serve(host: str = "127.0.0.1", port: int = 5000):
    """Start inference server.

    Args:
        host: Host to bind to
        port: Port to bind to
    """
    import uvicorn

    print("=" * 80)
    print("Starting Crypto DQN Inference Server")
    print("=" * 80)
    print("Server: http://{}:{}".format(host, port))
    print("API docs: http://{}:{}/docs".format(host, port))
    print("Health check: http://{}:{}/health".format(host, port))
    print("=" * 80)

    uvicorn.run("crypto_dqn_ops.inference.server:app", host=host, port=port, reload=False)


def main():
    """Main entry point for CLI."""
    fire.Fire(
        {
            "train": train,
            "infer": infer,
            "export_onnx": export_onnx,
            "serve": serve,
        }
    )


if __name__ == "__main__":
    main()
