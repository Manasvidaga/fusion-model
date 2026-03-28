# FNN + CNN Fusion — Fashion MNIST

An interactive machine learning project that trains, evaluates, and visualises three neural network architectures on the Fashion MNIST dataset: a Feedforward Neural Network (FNN), a Convolutional Neural Network (CNN), and a hybrid Fusion model that combines both branches for higher accuracy.

---

## Project Structure

```
.
├── index.html          # Landing page — links to all three simulations
├── fnn.html            # Interactive FNN forward-pass simulation
├── cnn.html            # Interactive CNN forward-pass simulation
├── fnn_cnn.html        # Interactive FNN + CNN Fusion simulation
└── notebooks
    ├── FNN.ipynb       # FNN training, evaluation & metrics
    ├── CNN.ipynb       # CNN training, evaluation & metrics
    └── fnn-cnn.ipynb   # Fusion model training, evaluation & metrics
```

## Dataset

**Fashion MNIST** — a drop-in replacement for MNIST using grayscale clothing images.

| Split | Size |
|---|---|
| Training | 60,000 images |
| Test | 10,000 images |
| Image size | 28 × 28 pixels, grayscale |
| Classes | 10 |

**Classes:** T-shirt/Top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle Boot

## Models

### 1. FNN — Feedforward Neural Network (`notebooks/FNN.ipynb`)

A standard multi-layer perceptron. The raw 28×28 image is flattened to a 784-dimensional vector and passed through fully-connected layers.

**Architecture**

```
Input (784)
  → Linear(784 → 256) → ReLU → Dropout(0.5)
  → Linear(256 → 128) → ReLU → Dropout(0.5)
  → Linear(128 → 64)  → ReLU → Dropout(0.5)
  → Linear(64  → 10)  → LogSoftmax
```

**Training config**

| Setting | Value |
|---|---|
| Optimizer | SGD (lr=0.001, momentum=0.9, weight_decay=1e-4) |
| Loss | NLLLoss |
| Batch size | 64 |
| Max epochs | 100 |
| Early stopping patience | 10 |
| Val split | 20% of training set |

**Strengths:** simple, fast to train, learns global relationships across all pixels.
**Limitations:** flattening destroys spatial structure; no understanding of pixel position or local patterns.

### 2. CNN — Convolutional Neural Network (`notebooks/CNN.ipynb`)

A deep convolutional network with three double-conv blocks, batch normalisation, max-pooling, and dropout.

**Architecture**

```
Input (1 × 28 × 28)
  → Conv(1→32) → BN → ReLU → Conv(32→32) → BN → ReLU → MaxPool → Dropout(0.20)   # 28→14
  → Conv(32→64) → BN → ReLU → Conv(64→64) → BN → ReLU → MaxPool → Dropout(0.25)  # 14→7
  → Conv(64→128) → BN → ReLU → ...                                                # 7→3
  → AdaptiveAvgPool → Flatten
  → FC → BN → ReLU → Dropout
  → Linear(→ 10)
```

**Training config**

| Setting | Value |
|---|---|
| Optimizer | Adam (lr=5e-4, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Loss | CrossEntropyLoss |
| Batch size | 64 |
| Max epochs | 50 |
| Early stopping patience | 15 |
| Augmentation | RandomCrop, RandomHorizontalFlip, RandomRotation(15°) |

**Strengths:** detects spatial features (edges, shapes, textures); parameter-efficient via kernel sharing.
**Limitations:** can miss global relationships; more complex to tune; higher compute cost.

### 3. FNN + CNN Fusion (`notebooks/fnn-cnn.ipynb`)

A dual-branch model that processes each image through both a residual CNN branch and an FNN branch in parallel, concatenates their feature vectors, and passes the combined representation through a joint classifier.

**Architecture**

```
Input (1 × 28 × 28)
     │
     ├── CNN Branch (Residual) ──────────────────────────────────┐
     │   Conv(1→32) → BN → ReLU                                 │
     │   ResBlock(32→64,   stride=2) → 14×14                    │
     │   ResBlock(64→128,  stride=2) →  7×7                     │
     │   ResBlock(128→256, stride=2) →  4×4                     │
     │   GlobalAvgPool → Flatten → FC(256→512) → 512-D          │
     │                                                           │
     └── FNN Branch ─────────────────────────────────────────────┤
         Flatten(784)                                            │
         FC(784→512) → BN → ReLU → Dropout(0.4)                │
         FC(512→256) → BN → ReLU → Dropout(0.3)                │
         FC(256→128) → BN → ReLU → 128-D                       │
                                                                 │
     Concat [512 + 128 = 640-D] ◄──────────────────────────────-┘
     FC(640→256) → BN → ReLU → Dropout(0.3)
     FC(256→10)  → Softmax
```

**Training config**

| Setting | Value |
|---|---|
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=80, eta_min=1e-5) |
| Loss | CrossEntropyLoss (label_smoothing=0.1) |
| Batch size | 64 |
| Max epochs | 80 |
| Early stopping patience | 15 |
| Gradient clipping | max_norm=1.0 |
| Augmentation | RandomHorizontalFlip, RandomCrop, RandomRotation(10°), ColorJitter |

**Why fusion?** The CNN branch captures local spatial patterns — edges, textures, shapes — that the FNN misses because flattening destroys pixel locality. The FNN branch captures global pixel relationships that a purely local convolution can overlook. Concatenating both 512-D and 128-D representations gives the classifier a 640-dimensional feature space that is richer than either model alone.

## Results

| Model | Test Accuracy |
|---|---|
| FNN | ~88% |
| CNN | ~91% |
| **FNN + CNN Fusion** | **~94%** |

All models are evaluated on the standard 10,000-sample Fashion MNIST test set. Metrics reported per model: accuracy, macro precision, macro recall, macro F1-score, and per-class confusion matrix.

### Model Inference on Raspberry Pi 5

![Inference Result](/fusion-model/assets/inference_res.png)

## Interactive Simulations

The HTML pages provide a browser-based, step-by-step animated visualisation of the forward pass for each architecture. Everything runs client-side — no server, no install.

### `fnn.html` — FNN Simulation

- Select any of the 10 Fashion MNIST classes; the 28×28 pixel image updates immediately
- Architecture builder: number of hidden layers (1–5), per-layer neuron count slider, activation function (ReLU / Tanh / Sigmoid), dropout rate, batch norm toggle
- Animated forward pass lights up each layer sequentially; neuron brightness reflects actual computed activation values, not random values
- Network diagram: nodes sized and coloured by activation magnitude; animated signal dots travel along connections between layers
- Output layer: all 10 class probabilities ranked with labelled bars, decimal percentages, and a confidence gauge

### `cnn.html` — CNN Simulation

- Same class selector and pixel image preview with hover-to-inspect coordinates
- Step-by-step forward pass: input → Conv Layer 1 (16 feature maps, teal colourmap) → Conv Layer 2 (32 feature maps) → MaxPool → Flatten + FC(128) → Output
- Feature map thumbnails render real simulated activations per filter, glowing on high-activation maps
- FC feature vector rendered as a horizontal pixel intensity bar
- Full 10-class probability output sorted by confidence with rank labels

### `fnn_cnn.html` — Fusion Simulation

- Both branches run in parallel with distinct colour coding: orange for FNN, teal for CNN, purple for Fusion/Output
- Fusion architecture diagram animates all layers including the final output layer — output nodes are sized proportionally to their real softmax probabilities and labelled with class names and percentages
- Every node in the diagram carries its actual activation weight derived from the selected input, not random values — changing the class changes every node's brightness across all layers
- Gating visualisation shows FNN and CNN feature contribution weights
- Comparison strip at the top shows all three model confidences side by side after each stream completes

## Getting Started

### Running the notebooks

```bash
pip install torch torchvision scikit-learn matplotlib seaborn numpy
jupyter notebook notebooks/FNN.ipynb
```

Fashion MNIST downloads automatically on first run via `torchvision.datasets.FashionMNIST`. A CUDA-capable GPU is recommended for the CNN and Fusion notebooks; the FNN notebook trains comfortably on CPU.

### Running the simulations

Open any HTML file directly in a browser — no build step needed:

```
open index.html
```

Or serve locally if you hit CORS issues:

```bash
python -m http.server 8000
# visit http://localhost:8000
```

## Dependencies

**Notebooks**

| Package | Purpose |
|---|---|
| `torch` | Model definition, training loop, inference |
| `torchvision` | FashionMNIST download and data augmentation |
| `scikit-learn` | Train/val split, accuracy, precision, recall, F1, confusion matrix |
| `matplotlib` | Training curves, prediction plots |
| `seaborn` | Confusion matrix heatmaps |
| `numpy` | Array operations |

**HTML simulations** — no dependencies. All logic is vanilla JavaScript. Fonts (Space Mono, Outfit) loaded from Google Fonts.

## Authors

| Roll Number | Name |
|---|---|
| AV.SC.U4AIE24003 | Aditya Anil |
| AV.SC.U4AIE24016 | Gishnu Gompa |
| AV.SC.U4AIE24019 | J. Pranathi |
| AV.SC.U4AIE24036 | Manasvi Daga |
| AV.SC.U4AIE24053 | Shashwat Mishra |
