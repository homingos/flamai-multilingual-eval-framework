"""
src/workers/inference.py
========================
VLLMWorker — parameterised vLLM inference worker.

One instance of this class is deployed per GPU tier. The model is identified
by its registry model_id (e.g. "tamil-mistral-7b"). All config is read
from the model's registry entry at container startup.

Modal volumes (mounted in modal_app.py):
  /data/benchmarks  — read-only dataset source
  /data/weights     — HuggingFace model cache (persists across runs)
  /data/outputs     — all output JSONL files

Critical constraint: this file must NOT import from src/.
The vllm_image does not have the registry package installed.
Use _ModelInfo dataclass and raw urllib.request calls to the registry API.
"""
from __future__ import annotations

from dataclasses import dataclass

import modal


# ---------------------------------------------------------------------------
# _ModelInfo — local stand-in for ModelConfig (no src/ import)
# ---------------------------------------------------------------------------

@dataclass
class _ModelInfo:
    id: str
    name: str
    hf_model_id: str
    language: str
    slug: str
    dtype: str
    gpu_memory_utilization: float
    max_model_len: int
    # None  → no template; use plain-text prompts via llm.generate()
    # str   → Jinja2 template; use structured chat via llm.chat()
    chat_template: str | None = None


# ---------------------------------------------------------------------------
# VLLMWorker
# ---------------------------------------------------------------------------

class VLLMWorker:
    """
    Parameterised vLLM inference worker.

    Subclassed in modal_app.py with @app.cls() and the appropriate GPU tier.
    Model loading happens once in @modal.enter() — not per-request.
    Output is written to the outputs volume incrementally as prompts complete.
    Checkpoint is written every CHECKPOINT_EVERY prompts for resume support.
    """

    # model_id, run_id, task are declared as modal.parameter() on each
    # @app.cls() subclass in modal_app.py — Modal injects them as instance
    # attributes before @modal.enter() is called.

    CHECKPOINT_EVERY = 50
    BATCH_SIZE       = 16
    TEMPERATURE      = 0.0
    TOP_P            = 1.0
    MAX_NEW_TOKENS   = 512

    @modal.enter()
    def load_model(self) -> None:
        """
        Runs once when the container starts (snapshotted — see modal_app.py).
        1. Sets HF_HOME so weights cache on the weights volume.
        2. Fetches ModelConfig from the registry API (includes chat_template).
        3. Loads vLLM LLM instance. If the model has a chat template stored in
           the registry, it is passed to llm.chat() at inference time.
           Falls back to plain-text llm.generate() if None.
        """
        import os
        os.environ["HF_HOME"] = "/data/weights"

        self.config = self._fetch_model_config()

        from vllm import LLM

        llm_kwargs: dict = dict(
            model=self.config.hf_model_id,
            dtype=self.config.dtype,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            trust_remote_code=True,
            disable_log_stats=True,
        )

        if self.config.chat_template:
            # Template will be passed to llm.chat() at inference time, not here.
            print(
                f"[VLLMWorker] {self.config.name}: using registry chat template "
                f"({len(self.config.chat_template)} chars)"
            )
        else:
            print(
                f"[VLLMWorker] {self.config.name}: no chat template in registry — "
                f"using plain-text prompt format"
            )

        self.llm = LLM(**llm_kwargs)

    def _fetch_model_config(self) -> _ModelInfo:
        """
        Reads model config directly from /data/registry/models.json (the
        phase2a-registry volume mounted by modal_app.py). No HTTP call needed.
        """
        import json

        registry_path = "/data/registry/models.json"
        try:
            with open(registry_path) as f:
                models = json.load(f)
        except FileNotFoundError:
            raise RuntimeError(
                f"Registry file not found at {registry_path}. "
                f"Ensure the phase2a-registry volume is mounted."
            )

        for data in models:
            if data.get("id") == self.model_id:
                return _ModelInfo(
                    id=data["id"],
                    name=data["name"],
                    hf_model_id=data["hf_model_id"],
                    language=data["language"],
                    slug=data["slug"],
                    dtype=data.get("dtype", "bfloat16"),
                    gpu_memory_utilization=data.get("gpu_memory_utilization", 0.88),
                    max_model_len=data.get("max_model_len", 2048),
                    chat_template=data.get("chat_template"),
                )

        raise RuntimeError(
            f"Model '{self.model_id}' not found in {registry_path}. "
            f"Add an entry with id='{self.model_id}' to models.json and re-upload."
        )

    @modal.method()
    def generate(self, samples: list) -> list:
        """
        Runs inference on a list of samples. Handles checkpoint/resume.

        Args:
            samples: list of raw sample dicts from the dataset

        Returns:
            list of output record dicts written during this call (excludes
            already-completed prompts that were skipped on resume)

        Steps:
        1. Determine output path for this model + task
        2. Load completed prompt IDs from existing output file (resume support)
        3. Filter out already-completed samples
        4. If nothing left, return []
        5. Build prompts, run vLLM in batches, write output records and checkpoints
        """
        from src.pipeline.run import (
            gemma_output_path,
            regional_output_path,
            load_completed_ids,
            append_output,
            write_checkpoint,
        )
        from src.pipeline.loader import build_prompt
        from vllm import SamplingParams
        from datetime import datetime, timezone

        # 1. Determine output path
        if self.model_id == "gemma-4-26b":
            out_path = gemma_output_path(self.run_id, self.task)
        else:
            out_path = regional_output_path(self.run_id, self.config.slug, self.task)

        # 2. Resume: skip already-completed prompts
        completed_ids = load_completed_ids(out_path)
        pending_samples = [s for s in samples if s["id"] not in completed_ids]

        if not pending_samples:
            return []

        # 3. Inference setup
        sampling_params = SamplingParams(
            temperature=self.TEMPERATURE,
            top_p=self.TOP_P,
            max_tokens=self.MAX_NEW_TOKENS,
        )

        def _make_record(sample: dict, output_text: str) -> dict:
            base = {
                "prompt_id":    sample["id"],
                "run_id":       self.run_id,
                "model_id":     self.model_id,
                "task":         self.task,
                "output":       output_text,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            if self.task == "translation":
                base.update({
                    "direction":  sample["direction"],
                    "source":     sample["source"],
                    "reference":  sample["reference"],
                })
            else:
                base.update({
                    "category":             sample["category"],
                    "system_instruction":   sample["system_instruction"],
                    "user_prompt":          sample.get("user_prompt_localized") or sample["user_prompt"],
                    "expected_constraints": sample["expected_constraints"],
                })
            return base

        # 4. Run in batches — use chat() if template exists, generate() otherwise
        results = []
        total_completed = len(completed_ids)

        for i in range(0, len(pending_samples), self.BATCH_SIZE):
            batch = pending_samples[i : i + self.BATCH_SIZE]

            if self.config.chat_template is not None:
                # ── Chat mode ────────────────────────────────────────────────
                # "tokenizer" sentinel → pass chat_template=None to llm.chat() so
                # vLLM uses the model's own built-in tokenizer template.
                # Any other string → use that Jinja2 template verbatim.
                conversations = []
                for sample in batch:
                    system_prompt, user_prompt = build_prompt(self.task, sample)
                    conversations.append([
                        {"role": "system",  "content": system_prompt},
                        {"role": "user",    "content": user_prompt},
                    ])
                outputs = self._safe_chat(conversations, sampling_params)
            else:
                # ── Plain-text mode ───────────────────────────────────────────
                # Model has no chat template — concatenate system + user with a
                # double newline. Works for base/pretrain models that don't
                # support structured chat (e.g. Tamil-Mistral base variant).
                prompts = []
                for sample in batch:
                    system_prompt, user_prompt = build_prompt(self.task, sample)
                    prompts.append(f"{system_prompt}\n\n{user_prompt}")
                outputs = self._safe_generate(prompts, sampling_params)

            for sample, output in zip(batch, outputs):
                if output is None:
                    continue
                output_text = output.outputs[0].text
                if "</think>" in output_text:
                    output_text = output_text.split("</think>", 1)[-1].strip()
                record = _make_record(sample, output_text)
                append_output(out_path, record)
                results.append(record)
                total_completed += 1

                if total_completed % self.CHECKPOINT_EVERY == 0:
                    write_checkpoint(
                        self.run_id, self.model_id, self.task,
                        sample["id"], total_completed,
                    )

        modal.Volume.from_name("phase2a-outputs").commit()
        return results

    def _safe_chat(self, conversations: list, sampling_params) -> list:
        """
        Wraps llm.chat() — used when a chat template is stored in the registry.
        The template is passed directly to chat() at inference time, which is
        the correct vLLM API (tokenizer_chat_template is not an LLM() constructor arg).
        Returns list of outputs, or list of None on error.
        """
        try:
            template = None if self.config.chat_template == "tokenizer" else self.config.chat_template
            return self.llm.chat(
                conversations,
                sampling_params=sampling_params,
                chat_template=template,
            )
        except Exception as exc:
            print(
                f"[VLLMWorker] chat() failed ({len(conversations)} conversations): {exc}"
            )
            return [None] * len(conversations)

    def _safe_generate(self, prompts: list, sampling_params) -> list:
        """
        Wraps llm.generate() — used when no chat template is stored in registry.
        Prompts are plain-text strings (system + user concatenated with newlines).
        Returns list of outputs, or list of None on error.
        """
        try:
            return self.llm.generate(prompts, sampling_params=sampling_params)
        except Exception as exc:
            print(f"[VLLMWorker] generate() failed ({len(prompts)} prompts): {exc}")
            return [None] * len(prompts)