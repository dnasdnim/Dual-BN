# Text to Bayesian Network framework

This folder collects the source code for preparing PRISM Bayesian network data, supervised fine-tuning (SFT), GRPO fine-tuning, checkpoint evaluation, and a separate backbone reconstruction → text → graph-agreement evaluation. The five requested entry points are copied from `training_models`; the three additional Python files are local imports needed by the backbone workflow. No datasets, model weights, generated results, or credentials are included.

## Files and workflow

| Stage | Entry point | Purpose |
| --- | --- | --- |
| 1. Prepare data | `preparing_training_data.py` | Read `prism_bn.json`, split by backbone, and save Hugging Face datasets in `data/{train,val,test}`. Rows cover four extraction passes and graph-to-text generation. |
| 2. SFT | `qwen14b_phase1.py` | Train a QLoRA adapter for `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` on the prepared training and validation rows. |
| 3. GRPO | `phase2_grpo.py` | Continue from the SFT adapter using structural, cycle, and graph-complexity rewards. |
| 4. Evaluate | `eval_checkpoint_qwen.py` | Run direct text-to-BN inference or held-out evaluation with node, state, edge, and CPD metrics. |
| 5. Backbone round trip | `evaluate_gpt56_backbone_text_roundtrip.py` | Reconstruct backbones with GPT-5.6, generate text from them, and judge agreement against the original graphs. This is a separate API-based workflow, not a required postprocessing step for the Qwen checkpoint. |

`evaluate_maximal_subgraph_union.py`, `evaluate_maximal_subgraph_union_prompted.py`, and `train_backbone_grpo.py` are included because stage 5 imports them. Keep these files beside its entry point.

## Setup

Run commands from this folder. Provide the source files locally before running:

- `prism_bn.json` for stages 1, 3, and held-out stage 4 evaluation.
- `prism_bn_openai.json` and `bn_marginal_raw_openai.jsonl` for the default stage 5 inputs. The JSONL needs the source article text used for text demonstrations. Use `--prism-json` and `--backbone-jsonl` to select other compatible files.

The scripts use Python packages including `torch`, `datasets`, `transformers`, `peft`, `trl`, `huggingface_hub`, `bitsandbytes`, and `openai`. Install versions compatible with your GPU and Transformers/TRL environment. Model downloads and the remote judges require access configured through runtime environment variables: `HF_TOKEN` for the Hugging Face judge and gated model access, and `OPENAI_API_KEY` for stage 5. Do not put tokens in source files.

## Example commands

These are examples only; they have not been executed as part of assembling this folder. SFT and GRPO require substantial GPU resources.

```bash
cd text_to_bn_framework
python preparing_training_data.py
python qwen14b_phase1.py
python phase2_grpo.py --adapter-path outputs/phase1_q1/final --prism-json prism_bn.json --train-dataset data/train --output-dir outputs/phase2_grpo
python eval_checkpoint_qwen.py --checkpoint outputs/phase2_grpo/checkpoint-200 --load-in-4bit
python evaluate_gpt56_backbone_text_roundtrip.py --prism-json prism_bn_openai.json --backbone-jsonl bn_marginal_raw_openai.jsonl --max-cases 5
```

The GRPO command uses the script's current default `--max-steps 200`; adjust it and the checkpoint path for the intended run. Stage 4's held-out mode expects `data/test` and `prism_bn.json` in this folder. For direct inference on one text file, pass `--input-file path/to/article.txt` with `--checkpoint`; that mode does not use the remote judge.

The stage 5 defaults use three demonstration backbone IDs (`bn_00004`, `bn_00006`, `bn_00027`). Supply compatible data containing those IDs, or choose IDs with repeated `--demo-id` arguments. Stage 5 calls the OpenAI API for backbone reconstruction, text generation, and judging; review its model and budget options before running it.

## Portability changes in these copies

- Data preparation now writes `data/`, matching the SFT, GRPO, and evaluation readers (the original script writes `data2/`).
- GRPO defaults to the SFT script's saved `outputs/phase1_q1/final` adapter and a relative `prism_bn.json` path.
- Checkpoint evaluation defaults to `outputs/phase1_q1/final` within this folder.

All other research logic remains copied from the original scripts. Large input corpora, dataset splits, checkpoints, and generated outputs are excluded by `.gitignore` so the folder can be added to a GitHub repository as source code.
