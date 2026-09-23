"""
Method names to look for in an abstract.

Grouped by family, deduplicated, and matched longest first so a paper that says
"Convolutional Neural Network" is reported as that and not as the "CNN" sitting
inside the phrase. Frameworks are deliberately absent: matching "PyTorch" tells
you what someone built with, not what they built.
"""

from __future__ import annotations

import re

ARCHITECTURES: tuple[str, ...] = (
    "Multilayer Perceptron", "MLP",
    "Convolutional Neural Network", "CNN",
    "Recurrent Neural Network", "RNN",
    "Long Short-Term Memory", "LSTM",
    "Gated Recurrent Unit", "GRU",
    "Graph Neural Network", "GNN",
    "Graph Convolutional Network", "GCN",
    "Message Passing Neural Network", "MPNN",
    "Deep Belief Network",
    "Radial Basis Function Network",
    "Autoencoder", "Variational Autoencoder",
    "Generative Adversarial Network", "GAN",
    "Capsule Network",
    "Residual Network", "ResNet",
    "U-Net", "DenseNet", "AlexNet", "VGG", "LeNet", "GoogLeNet", "SqueezeNet",
    "Inception Network", "MobileNet", "EfficientNet",
    "Vision Transformer", "ViT",
    "Transformer",
)

LANGUAGE_MODELS: tuple[str, ...] = (
    "BERT", "BioBERT", "SciBERT", "PubMedBERT", "ClinicalBERT",
    "RoBERTa", "DeBERTa", "ALBERT", "ELECTRA", "XLNet", "ERNIE", "T5",
    "GPT-2", "GPT-3", "GPT-4", "GPT",
    "LLaMA", "Mistral", "Claude", "Gemini",
    "Word2Vec", "Doc2Vec", "FastText", "GloVe", "Mol2vec",
)

CLASSICAL_ML: tuple[str, ...] = (
    "Support Vector Machine", "SVM",
    "Support Vector Regression", "SVR",
    "Random Forest", "Extra Trees", "Decision Tree",
    "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost", "AdaBoost",
    "Logistic Regression", "Linear Regression", "Ridge Regression", "Lasso",
    "K-Nearest Neighbors", "KNN",
    "Naive Bayes", "Perceptron",
    "Partial Least Squares", "PLS",
    "Gaussian Process",
)

UNSUPERVISED: tuple[str, ...] = (
    "Principal Component Analysis", "PCA",
    "K-Means", "DBSCAN", "Mean-Shift",
    "Hierarchical Clustering",
    "Self-Organizing Map",
    "t-SNE", "UMAP",
    "Hidden Markov Model", "Markov Model",
)

REINFORCEMENT: tuple[str, ...] = (
    "Deep Q-Network", "DQN",
    "Proximal Policy Optimization", "PPO",
    "Soft Actor-Critic", "SAC",
    "Q-Learning", "SARSA",
    "Policy Gradient",
    "Reinforcement Learning",
)

DOMAIN_SPECIFIC: tuple[str, ...] = (
    "DeePred-BBB", "LogBB_Pred", "Deep-B3",
    "QSAR", "QSPR",
    "Molecular Docking",
    "Random Forest Regressor",
)

GROUPS: dict[str, tuple[str, ...]] = {
    "architecture": ARCHITECTURES,
    "language_model": LANGUAGE_MODELS,
    "classical_ml": CLASSICAL_ML,
    "unsupervised": UNSUPERVISED,
    "reinforcement": REINFORCEMENT,
    "domain": DOMAIN_SPECIFIC,
}


def _unique(terms: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for term in terms:
        key = term.casefold()
        if key not in seen:
            seen.add(key)
            out.append(term)
    return out


ALL_TERMS: list[str] = _unique(tuple(term for group in GROUPS.values() for term in group))

GROUP_OF: dict[str, str] = {
    term.casefold(): group for group, terms in GROUPS.items() for term in terms
}

# Longest first, so "Convolutional Neural Network" is preferred over "CNN", and
# word boundaries so "GAN" does not match inside "organ".
_PATTERN = re.compile(
    r"(?<![\w-])(" + "|".join(re.escape(t) for t in sorted(ALL_TERMS, key=len, reverse=True)) + r")(?![\w-])",
    re.IGNORECASE,
)


def find_terms(text: str) -> list[str]:
    """Every known method named in `text`, canonically cased, in first-seen order."""
    if not text:
        return []

    canonical = {term.casefold(): term for term in ALL_TERMS}
    found: list[str] = []
    seen: set[str] = set()
    covered: list[tuple[int, int]] = []

    for match in _PATTERN.finditer(text):
        start, end = match.span()
        # Skip a hit that sits inside a longer one already taken, so
        # "Convolutional Neural Network" does not also report "Network".
        if any(start >= s and end <= e for s, e in covered):
            continue
        covered.append((start, end))

        term = canonical[match.group(1).casefold()]
        if term not in seen:
            seen.add(term)
            found.append(term)

    return found
