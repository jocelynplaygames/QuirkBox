# 🧠 QuirkBox: A Lightweight, Visualizable Transformer Text Generation Platform

> ⚡️ Extended from [rasbt/llama-3.2-from-scratch](https://github.com/rasbt/llama-3.2-from-scratch)
> 🛠️ With enhancements for attention visualization, web UI, and configurable training.

---

## 🚀 Features

- ✅ Fully customizable Transformer model (`MiniGenieTransformer`)
- ✅ Switchable modules: LayerNorm, Dropout, Rotary Embeddings, Position Embeddings
- ✅ Exports attention weights for inspection and visualization
- ✅ Web UI with prompt-style selection & live generation
- ✅ Support for token embedding configuration
- ✅ Training-ready via Hydra config + Weights & Biases logging
- ✅ CLI + Web dual inference interface

---

## 📦 Installation

```bash

git clone https://github.com/yourusername/QuirkBox.git
cd QuirkBox
pip install -r requirements.txt

```

Make sure you also have:

```bash

pip install torch tiktoken blobfile hydra-core wandb

```

---

## 📁 Project Structure

```
QuirkBox/
├── model.py               # 🧠 Redesigned MiniGenieTransformer (based on LLaMA3)
├── generate.py            # 🔮 CLI generation with attention export
├── app.py                 # 🌐 Flask Web API for interaction
├── templates/index.html   # 🎨 Frontend UI + canvas attention viewer
├── static/attn.json       # 🧪 Saved attention weights for visualization
├── configs/               # ⚙️ Training config (Hydra)
├── tokenizer.py           # 🔤 Tokenizer setup (from original project)
├── weights/               # 📦 Downloaded model weights (.pth)
├── README.md
└── LICENSE

```

---

## 💻 Usage: CLI Mode

Run inference from CLI:

```bash
python generate.py --input "Why do cats meow?" --style casual

```

Options:

- `-style`:
    - `casual`, `joke`, `sci-fi`, `poetry`, `therapy`
- `-output-attn`: export attention weights to `attn.json` for frontend visualization

---

## 🌐 Usage: Web UI

Launch the web interface:

```bash
python app.py

```

In browser:

- Select prompt style
- Enter message
- View response + attention map (layer/head selectable)

---

## ⚙️ Training & Experimentation

This repo includes **Hydra config + wandb logging** to simulate training:

```bash
python train.py experiment=demo

```

Outputs:

- Model checkpoints
- `wandb` metrics
- Configurable architecture (LayerNorm, Dropout, Embedding, etc.)

---

## 🎯 What Makes This Project Unique?

| Feature | Description |
| --- | --- |
| ✅ Attention Export | Returns attention matrix from each layer/head |
| ✅ Visual Debugging | Canvas UI lets user view where model is focusing |
| ✅ Model Architecture | LayerNorm, Dropout, PosEmbed fully configurable |
| ✅ Modularity | Easy to add new attention types or layers |
| ✅ Training Ready | Configurable & extensible for small-scale training |

---

## Credits & Attribution

This project builds upon:

- [rasbt/llama-3.2-from-scratch](https://github.com/rasbt/llama-3.2-from-scratch)
    
    A minimal, educational PyTorch implementation of LLaMA 3.2
