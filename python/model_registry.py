"""
Model Registry — Versioning, comparison, and hot-swap of trained models.

Workflow:
  1. Offline Trainer saves new model → Registry
  2. Registry evaluates model on validation metric
  3. If better than current best → promote to "active"
  4. Online Scheduler hot-swaps to new active model

Supports: autoformer, bilstm, agent1-4, central_critic
"""

import os
import json
import shutil
import torch
import threading
from datetime import datetime
from config import Config


class ModelRegistry:
    """Thread-safe model registry with versioning and best-model selection."""

    VALID_MODEL_TYPES = (
        "autoformer", "bilstm", "agent1", "agent2",
        "agent3", "agent4", "central_critic"
    )

    def __init__(self, model_dir=None):
        self.model_dir = model_dir or Config.MODEL_DIR
        os.makedirs(self.model_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._versions = {}  # model_type -> list of version dicts
        self._active = {}    # model_type -> version_id
        self._load_index()

    def _index_path(self):
        return os.path.join(self.model_dir, "registry_index.json")

    def _load_index(self):
        """Load registry index from disk."""
        path = self._index_path()
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                self._versions = data.get("versions", {})
                self._active = data.get("active", {})

    def _save_index(self):
        """Persist registry index to disk."""
        data = {"versions": self._versions, "active": self._active}
        with open(self._index_path(), "w") as f:
            json.dump(data, f, indent=2)

    def save_model(self, model, model_type, metrics=None, event_db=None):
        """
        Save a model checkpoint and register it.

        Args:
            model: nn.Module or state_dict
            model_type: one of VALID_MODEL_TYPES
            metrics: dict with evaluation metrics (must include val_loss)
            event_db: optional EventDatabase to log model_swap

        Returns:
            version_id: int
        """
        assert model_type in self.VALID_MODEL_TYPES, \
            f"Unknown model_type: {model_type}"

        with self._lock:
            # Determine version number
            versions = self._versions.get(model_type, [])
            version_id = len(versions) + 1

            # Save model file
            filename = f"{model_type}_v{version_id}.pt"
            filepath = os.path.join(self.model_dir, filename)

            if isinstance(model, dict):
                torch.save(model, filepath)
            else:
                torch.save(model.state_dict(), filepath)

            # Register
            entry = {
                "version": version_id,
                "path": filepath,
                "filename": filename,
                "metrics": metrics or {},
                "val_loss": metrics.get("val_loss", float("inf")) if metrics else float("inf"),
                "created_at": datetime.now().isoformat(),
            }

            if model_type not in self._versions:
                self._versions[model_type] = []
            self._versions[model_type].append(entry)

            # Auto-promote if better than current best
            promoted = False
            current_best = self.get_best_version_info(model_type)
            if current_best is None or entry["val_loss"] < current_best["val_loss"]:
                self._active[model_type] = version_id
                promoted = True

            self._save_index()

        if promoted:
            print(f"[Registry] {model_type} v{version_id} PROMOTED as best "
                  f"(val_loss={entry['val_loss']:.6f})")
        else:
            print(f"[Registry] {model_type} v{version_id} saved "
                  f"(val_loss={entry['val_loss']:.6f}, "
                  f"best remains v{self._active.get(model_type, '?')})")

        return version_id

    def get_best_model_path(self, model_type):
        """Get filepath of the currently active (best) model."""
        with self._lock:
            active_ver = self._active.get(model_type)
            if active_ver is None:
                return None
            versions = self._versions.get(model_type, [])
            for v in versions:
                if v["version"] == active_ver:
                    return v["path"]
            return None

    def load_best_model(self, model_type, model_class=None, **model_kwargs):
        """
        Load the best model for a given type.

        Args:
            model_type: one of VALID_MODEL_TYPES
            model_class: nn.Module class to instantiate
            **model_kwargs: kwargs for model_class constructor

        Returns:
            state_dict if model_class is None, else instantiated model
        """
        path = self.get_best_model_path(model_type)
        if path is None or not os.path.exists(path):
            return None

        state_dict = torch.load(path, map_location="cpu", weights_only=True)

        if model_class is not None:
            model = model_class(**model_kwargs)
            model.load_state_dict(state_dict)
            model.eval()
            return model

        return state_dict

    def get_best_version_info(self, model_type):
        """Get metadata of the current best version."""
        active_ver = self._active.get(model_type)
        if active_ver is None:
            return None
        versions = self._versions.get(model_type, [])
        for v in versions:
            if v["version"] == active_ver:
                return v
        return None

    def get_active_version(self, model_type):
        """Get the active version number."""
        return self._active.get(model_type)

    def list_versions(self, model_type):
        """List all versions for a model type."""
        return self._versions.get(model_type, [])

    def has_model(self, model_type):
        """Check if any version exists for this model type."""
        return model_type in self._active and self._active[model_type] is not None

    def register_initial_model(self, model_type, filepath, metrics=None):
        """
        Register an existing model file (e.g., from baseline training).
        Used to seed the registry with pre-trained models.
        """
        assert os.path.exists(filepath), f"File not found: {filepath}"

        with self._lock:
            version_id = 1
            # Copy to registry directory
            dest = os.path.join(self.model_dir, f"{model_type}_v{version_id}.pt")
            if os.path.abspath(filepath) != os.path.abspath(dest):
                shutil.copy2(filepath, dest)

            entry = {
                "version": version_id,
                "path": dest,
                "filename": f"{model_type}_v{version_id}.pt",
                "metrics": metrics or {},
                "val_loss": metrics.get("val_loss", 0.0) if metrics else 0.0,
                "created_at": datetime.now().isoformat(),
            }
            self._versions[model_type] = [entry]
            self._active[model_type] = version_id
            self._save_index()

        print(f"[Registry] Registered initial {model_type} v1 from {filepath}")
        return version_id

    def summary(self):
        """Print registry summary."""
        print("\n=== Model Registry Summary ===")
        for mt in self.VALID_MODEL_TYPES:
            versions = self._versions.get(mt, [])
            active = self._active.get(mt)
            if versions:
                best_info = self.get_best_version_info(mt)
                print(f"  {mt}: {len(versions)} versions, "
                      f"active=v{active}, "
                      f"best_val_loss={best_info['val_loss']:.6f}")
            else:
                print(f"  {mt}: no versions")
        print("=" * 30)
