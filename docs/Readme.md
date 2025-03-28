# Deep Reinforcement Learning with SIMBA for SoccerTwos

This repository contains research implementing SIMBA (Simplicity Bias for Scaling Up Parameters) with reinforcement learning algorithms PPO (Proximal Policy Optimization) and SAC (Soft Actor-Critic) for Unity's SoccerTwos environment.

## Project Overview

The research aims to evaluate the impact of incorporating the SIMBA architecture into existing reinforcement learning algorithms (PPO and SAC) within the SoccerTwos game environment.

## Methods and Architecture

### Reinforcement Learning Algorithms

- **PPO**: An on-policy method optimizing stochastic policies using a clipped surrogate objective.
- **SAC**: An off-policy method optimizing stochastic policies with entropy regularization.

### SIMBA Architecture

SIMBA introduces simplicity bias to neural networks and includes:
- Observation normalization (RSNorm)
- Residual feedforward blocks
- Post-layer normalization

## Repository Structure

- `config/`: Configuration files for training.
- `ml-agents/mlagents/trainers/`: Python scripts for PPO and SAC, including SIMBA implementations.
- `Project/Assets/ML-Agents/Examples/Soccer/`: Modified Unity environment for SoccerTwos.

## Demonstration Videos

### PPO vs PPO_SIMBA
[Insert video here]

### SAC vs SAC_SIMBA
[Insert video here]

## Setup and Usage

Clone the repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run training:
```bash
python train.py --config=config/your-config.yaml
```

## Notes

- Results, pretrained models, and large files are intentionally excluded to maintain repository efficiency and reproducibility.
- Original Unity ML-Agents repository: [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents)

## Citation

If you find this research useful, please cite the original ML-Agents paper:

```
@article{juliani2018unity,
  title={Unity: A general platform for intelligent agents},
  author={Juliani, Arthur and Berges, Vincent-Pierre and Vckay, Esh and Gao, Yuan and Henry, Hunter and Mattar, Marwan and Lange, Danny},
  journal={arXiv preprint arXiv:1809.02627},
  year={2018}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

