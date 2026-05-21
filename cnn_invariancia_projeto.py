"""
Projeto: CNNs são realmente invariantes?
Script principal de treinamento e avaliação.

Uso:
    python cnn_invariancia_projeto.py

O script importa o arquivo de apoio `mnist_cnn_invariancia_augmentation.py`
(deve estar no mesmo diretório) para obter os dataloaders.

Requisitos:
    pip install torch torchvision matplotlib
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# Importa utilitários do script de apoio fornecido pelo professor
from mnist_cnn_invariancia_augmentation import (
    LoaderConfig,
    get_mnist_dataloaders,
    seed_everything,
)

# ---------------------------------------------------------------------------
# Configurações globais
# ---------------------------------------------------------------------------

SEED = 42
EPOCHS = 10
BATCH_SIZE = 128
LR = 1e-3
VAL_FRACTION = 0.1        # 10 % do treino reservado para validação
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("./resultados")
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Definição das arquiteturas
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """
    Arquitetura 1 – CNN pequena (baseline).

    Estrutura:
        Conv(1→16, 3×3) → ReLU → MaxPool(2×2)
        Conv(16→32, 3×3) → ReLU → MaxPool(2×2)
        Flatten → Linear(1568→64) → ReLU → Linear(64→10)

    Pergunta associada: Uma CNN pequena já aprende padrões robustos a
    transformações fora da distribuição de treino?
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # 28×28×16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # 14×14×16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 14×14×32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # 7×7×32
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class DeepCNN(nn.Module):
    """
    Arquitetura 2 – CNN mais profunda com Dropout.

    Estrutura:
        Bloco 1: Conv(1→32, 3×3) → BN → ReLU → Conv(32→32, 3×3) → BN → ReLU → MaxPool(2×2)
        Bloco 2: Conv(32→64, 3×3) → BN → ReLU → Conv(64→64, 3×3) → BN → ReLU → MaxPool(2×2)
        Bloco 3: Conv(64→128, 3×3) → BN → ReLU
        Flatten → Dropout(0.4) → Linear(128×5×5→256) → ReLU → Dropout(0.3) → Linear(256→10)

    Adições em relação à SimpleCNN:
        - Terceiro bloco convolucional (maior profundidade)
        - BatchNormalization após cada convolução (estabilidade de treino)
        - Dropout nas camadas densas (regularização)

    Perguntas associadas:
        - Mais profundidade melhora generalização ou apenas acurácia no teste original?
        - Dropout melhora robustez às transformações?
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.block1 = conv_block(1, 32)          # 28×28×32
        self.pool1 = nn.MaxPool2d(2)             # 14×14×32
        self.block2 = conv_block(32, 64)         # 14×14×64
        self.pool2 = nn.MaxPool2d(2)             # 7×7×64
        self.block3 = nn.Sequential(             # 5×5×128 (sem pool extra)
            nn.Conv2d(64, 128, kernel_size=3),   # 5×5×128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128 * 5 * 5, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.block3(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Funções de treino e avaliação
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> Tuple[float, float]:
    """Treina por uma época. Retorna (loss_média, acurácia_treino)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
    """Avalia o modelo. Retorna (loss_média, acurácia)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


def build_val_split(
    train_loader: DataLoader, val_fraction: float = VAL_FRACTION
) -> Tuple[DataLoader, DataLoader]:
    """
    Divide o dataset do train_loader em treino e validação.
    Mantém batch_size, num_workers e pin_memory originais.
    """
    dataset = train_loader.dataset
    n_val = int(len(dataset) * val_fraction)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    kwargs = dict(
        batch_size=train_loader.batch_size,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory,
    )
    return (
        DataLoader(train_ds, shuffle=True, **kwargs),
        DataLoader(val_ds, shuffle=False, **kwargs),
    )


# ---------------------------------------------------------------------------
# Loop de treinamento completo (com registro de métricas)
# ---------------------------------------------------------------------------

def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = EPOCHS,
    lr: float = LR,
    model_name: str = "model",
) -> Dict[str, List[float]]:
    """
    Treina o modelo e registra métricas por época.

    Retorna dicionário com chaves:
        train_loss, train_acc, val_loss, val_acc
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    print(f"\n{'='*60}")
    print(f"  Treinando: {model_name}  |  Device: {DEVICE}")
    print(f"{'='*60}")
    print(f"{'Época':>6} | {'L-treino':>9} | {'Acc-treino':>10} | {'L-val':>7} | {'Acc-val':>8}")
    print("-" * 60)

    best_val_acc = 0.0
    best_path = RESULTS_DIR / f"{model_name}_best.pt"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc = evaluate(model, val_loader)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        # Salva o melhor checkpoint
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), best_path)

        elapsed = time.time() - t0
        print(
            f"{epoch:>6} | {tr_loss:>9.4f} | {tr_acc:>10.4f} | "
            f"{vl_loss:>7.4f} | {vl_acc:>8.4f}  [{elapsed:.1f}s]"
        )

    # Carrega o melhor checkpoint ao final
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    print(f"\n  Melhor val_acc: {best_val_acc:.4f}  (checkpoint restaurado)")
    return history


# ---------------------------------------------------------------------------
# Avaliação final em todos os conjuntos de teste
# ---------------------------------------------------------------------------

def evaluate_all_tests(
    model: nn.Module, test_loaders: Dict[str, DataLoader]
) -> Dict[str, float]:
    """Avalia o modelo em todos os loaders de teste e retorna acurácias."""
    results = {}
    for name, loader in test_loaders.items():
        _, acc = evaluate(model, loader)
        results[name] = acc
    return results


# ---------------------------------------------------------------------------
# Visualizações
# ---------------------------------------------------------------------------

def plot_history(
    history: Dict[str, List[float]],
    title: str,
    save_path: Path,
) -> None:
    """Plota curvas de loss e acurácia de treino/validação."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Treino")
    axes[0].plot(epochs, history["val_loss"], label="Validação", linestyle="--")
    axes[0].set_title(f"{title} – Loss")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Acurácia
    axes[1].plot(epochs, history["train_acc"], label="Treino")
    axes[1].plot(epochs, history["val_acc"], label="Validação", linestyle="--")
    axes[1].set_title(f"{title} – Acurácia")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Acurácia")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico salvo: {save_path}")


def plot_results_table(
    all_results: Dict[str, Dict[str, float]],
    save_path: Path,
) -> None:
    """
    Plota tabela comparativa de acurácias como heatmap.
    Linhas = modelos, colunas = conjuntos de teste.
    """
    test_names = [
        "test_original", "test_translated", "test_rotated",
        "test_noisy", "test_contrast", "test_combined",
    ]
    col_labels = [n.replace("test_", "") for n in test_names]
    row_labels = list(all_results.keys())

    data = np.array(
        [[all_results[model].get(t, float("nan")) for t in test_names]
         for model in row_labels]
    )

    fig, ax = plt.subplots(figsize=(10, max(3, len(row_labels) * 0.8 + 1.5)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color="black")

    ax.set_title("Acurácia por modelo e conjunto de teste", fontsize=11)
    plt.colorbar(im, ax=ax, label="Acurácia")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Tabela salva: {save_path}")


def show_misclassified(
    model: nn.Module,
    loader: DataLoader,
    loader_name: str,
    n: int = 8,
    save_path: Path | None = None,
) -> None:
    """Exibe imagens classificadas incorretamente."""
    model.eval()
    wrong_images, wrong_preds, wrong_labels = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(1)
            mask = preds != y
            wrong_images.extend(x[mask].cpu())
            wrong_preds.extend(preds[mask].cpu().tolist())
            wrong_labels.extend(y[mask].cpu().tolist())
            if len(wrong_images) >= n:
                break

    n = min(n, len(wrong_images))
    if n == 0:
        print(f"  {loader_name}: nenhum erro encontrado!")
        return

    fig, axes = plt.subplots(1, n, figsize=(n * 1.6, 2))
    if n == 1:
        axes = [axes]
    for i in range(n):
        img = wrong_images[i].squeeze().numpy()
        # Desnormaliza aproximadamente para visualização
        img = img * 0.3081 + 0.1307
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(
            f"R:{wrong_labels[i]}\nP:{wrong_preds[i]}",
            fontsize=7, color="red"
        )
        axes[i].axis("off")
    plt.suptitle(f"Erros – {loader_name}", fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Imagens salvas: {save_path}")
    plt.close()


def visualize_first_layer_filters(
    model: nn.Module,
    model_name: str,
    save_path: Path,
) -> None:
    """Visualiza os filtros aprendidos na primeira camada convolucional."""
    # Encontra o primeiro módulo Conv2d
    first_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            first_conv = m
            break
    if first_conv is None:
        return

    weights = first_conv.weight.data.cpu()  # (out_ch, 1, H, W)
    n_filters = weights.shape[0]
    cols = min(8, n_filters)
    rows = (n_filters + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.3, rows * 1.3))
    axes = np.array(axes).reshape(-1)

    for i in range(n_filters):
        w = weights[i, 0]
        w = (w - w.min()) / (w.max() - w.min() + 1e-8)
        axes[i].imshow(w.numpy(), cmap="gray")
        axes[i].axis("off")
    for i in range(n_filters, len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"Filtros – 1ª camada conv ({model_name})", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Filtros salvos: {save_path}")


def visualize_activation_maps(
    model: nn.Module,
    loader: DataLoader,
    model_name: str,
    save_path: Path,
    n_images: int = 4,
) -> None:
    """
    Usa hooks do PyTorch para capturar e visualizar os mapas de ativação
    após a primeira camada ReLU.
    """
    activations: Dict[str, torch.Tensor] = {}

    def hook_fn(name: str):
        def fn(module, inp, out):
            activations[name] = out.detach().cpu()
        return fn

    # Registra hook na primeira ReLU
    handle = None
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            handle = module.register_forward_hook(hook_fn("relu1"))
            break

    model.eval()
    x, y = next(iter(loader))
    x_dev = x[:n_images].to(DEVICE)

    with torch.no_grad():
        _ = model(x_dev)

    if handle:
        handle.remove()

    if "relu1" not in activations:
        print("  Hook não capturou ativações.")
        return

    act = activations["relu1"]  # (n, C, H, W)
    n_channels = min(8, act.shape[1])

    fig, axes = plt.subplots(
        n_images, n_channels + 1,
        figsize=((n_channels + 1) * 1.4, n_images * 1.4)
    )

    for i in range(n_images):
        # Imagem original (desnormalizada)
        img = x[i].squeeze().numpy() * 0.3081 + 0.1307
        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].set_title(f"label={y[i].item()}", fontsize=7)
        axes[i, 0].axis("off")

        for j in range(n_channels):
            a = act[i, j].numpy()
            axes[i, j + 1].imshow(a, cmap="viridis")
            axes[i, j + 1].axis("off")
            if i == 0:
                axes[i, j + 1].set_title(f"ch{j}", fontsize=7)

    axes[0, 0].set_title("Original", fontsize=7)
    plt.suptitle(f"Mapas de ativação (1ª ReLU) – {model_name}", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Mapas de ativação salvos: {save_path}")


# ---------------------------------------------------------------------------
# Ponto de entrada principal
# ---------------------------------------------------------------------------

def main() -> None:
    seed_everything(SEED)

    # ------------------------------------------------------------------
    # 1. Carrega dataloaders do script de apoio
    # ------------------------------------------------------------------
    config = LoaderConfig(batch_size=BATCH_SIZE, num_workers=2, seed=SEED)
    train_loaders, test_loaders = get_mnist_dataloaders(config)

    # Cria splits de validação para cada condição de treino
    plain_train_loader, plain_val_loader = build_val_split(
        train_loaders["train_plain"]
    )
    aug_train_loader, aug_val_loader = build_val_split(
        train_loaders["train_augmented"]
    )

    # ------------------------------------------------------------------
    # 2. Define os quatro experimentos (arquitetura × condição de treino)
    # ------------------------------------------------------------------
    experiments = [
        ("SimpleCNN_plain",    SimpleCNN().to(DEVICE), plain_train_loader, plain_val_loader),
        ("SimpleCNN_augmented",SimpleCNN().to(DEVICE), aug_train_loader,   aug_val_loader),
        ("DeepCNN_plain",      DeepCNN().to(DEVICE),   plain_train_loader, plain_val_loader),
        ("DeepCNN_augmented",  DeepCNN().to(DEVICE),   aug_train_loader,   aug_val_loader),
    ]

    all_histories: Dict[str, Dict] = {}
    all_results: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # 3. Treina e avalia cada experimento
    # ------------------------------------------------------------------
    for name, model, tr_loader, vl_loader in experiments:
        seed_everything(SEED)  # Reprodutibilidade por experimento

        history = run_training(model, tr_loader, vl_loader, epochs=EPOCHS, model_name=name)
        all_histories[name] = history

        # Plota curvas de aprendizado
        plot_history(history, name, RESULTS_DIR / f"curvas_{name}.png")

        # Avalia em todos os conjuntos de teste
        results = evaluate_all_tests(model, test_loaders)
        all_results[name] = results

        # Salva imagens de erros (3 condições)
        for test_key in ["test_original", "test_rotated", "test_combined"]:
            show_misclassified(
                model, test_loaders[test_key],
                loader_name=f"{name}_{test_key}",
                n=8,
                save_path=RESULTS_DIR / f"erros_{name}_{test_key}.png",
            )

        # Visualizações qualitativas
        visualize_first_layer_filters(
            model, name, RESULTS_DIR / f"filtros_{name}.png"
        )
        visualize_activation_maps(
            model, test_loaders["test_original"],
            name, RESULTS_DIR / f"ativacoes_{name}.png"
        )

    # ------------------------------------------------------------------
    # 4. Tabela comparativa final
    # ------------------------------------------------------------------
    print("\n\n" + "="*70)
    print("TABELA COMPARATIVA – ACURÁCIA POR MODELO E CONJUNTO DE TESTE")
    print("="*70)

    test_keys = [
        "test_original", "test_translated", "test_rotated",
        "test_noisy", "test_contrast", "test_combined",
    ]
    header = f"{'Modelo':<26}" + "".join(f"{k.replace('test_',''):>12}" for k in test_keys)
    print(header)
    print("-" * len(header))

    for model_name, results in all_results.items():
        row = f"{model_name:<26}"
        for k in test_keys:
            row += f"{results.get(k, float('nan')):>12.4f}"
        print(row)

    # Salva JSON para uso no relatório
    with open(RESULTS_DIR / "resultados.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResultados salvos em {RESULTS_DIR / 'resultados.json'}")

    # Plota heatmap
    plot_results_table(all_results, RESULTS_DIR / "tabela_comparativa.png")

    print("\nTreinamento concluído. Verifique a pasta ./resultados/")


if __name__ == "__main__":
    main()
