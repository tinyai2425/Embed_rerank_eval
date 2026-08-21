#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared path helpers for case generation."""

import os


def repo_root() -> str:
    """Embed_rerank_eval repo root (parent of 1_Data_gen)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))


def data_set_dir() -> str:
    return os.path.join(repo_root(), "data_set")


def storage_root() -> str:
    """../model-eval-storage relative to this repo (Embed_rerank_eval)."""
    return os.path.join(os.path.dirname(repo_root()), "model-eval-storage")


def storage_base(model_name: str) -> str:
    return os.path.join(storage_root(), model_name)


def get_unique_subdir(base_dir: str, prefix: str = "project") -> str:
    counter = 1
    while os.path.exists(os.path.join(base_dir, f"{prefix}-{counter}")):
        counter += 1
    return f"{prefix}-{counter}"


def make_project_dir(model_name: str) -> tuple:
    """Create a new project-N directory under ../model-eval-storage/<model_name>."""
    base = storage_base(model_name)
    name = get_unique_subdir(base, "project")
    path = os.path.join(base, name)
    os.makedirs(path, exist_ok=True)
    return name, path
