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
        Runs once when the container starts.
        1. Sets HF_HOME so weights cache on the weights volume.
        2. Fetches ModelConfig from the registry API.
        3. Loads vLLM LLM instance from HuggingFace (or volume cache).
        """
        import os
        os.environ["HF_HOME"] = "/data/weights"

        self.config = self._fetch_model_config()

        from vllm import LLM
        self.llm = LLM(
            model=self.config.hf_model_id,
            dtype=self.config.dtype,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            trust_remote_code=True,
        )

    def _fetch_model_config(self) -> _ModelInfo:
        """
        Fetches model data from the registry API using urllib (no src/ import).
        Reads REGISTRY_URL and JWT_TOKEN from environment (set via Modal secrets).
        Raises RuntimeError if the model is not found or the request fails.
        """
        import json
        import os
        import urllib.request

        registry_url = os.environ["REGISTRY_URL"]
        jwt_token    = os.environ["JWT_TOKEN"]
        url = f"{registry_url}/models/{self.model_id}"

        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {jwt_token}"},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                body = json.loads(resp.read())
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch model config for '{self.model_id}': {exc}"
            ) from exc

        data = body.get("data")
        if data is None:
            raise RuntimeError(
                f"Registry returned no data for model '{self.model_id}'. "
                f"Response: {body}"
            )

        return _ModelInfo(
            id=data["id"],
            name=data["name"],
            hf_model_id=data["hf_model_id"],
            language=data["language"],
            slug=data["slug"],
            dtype=data.get("dtype", "bfloat16"),
            gpu_memory_utilization=data.get("gpu_memory_utilization", 0.88),
            max_model_len=data.get("max_model_len", 2048),
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
                    "user_prompt":          sample["user_prompt"],
                    "expected_constraints": sample["expected_constraints"],
                })
            return base

        # 4. Run in batches
        results = []
        total_completed = len(completed_ids)

        for i in range(0, len(pending_samples), self.BATCH_SIZE):
            batch = pending_samples[i : i + self.BATCH_SIZE]

            # Format as plain text — avoids chat-template issues for models
            # like Tamil-Mistral-7B that don't define a tokenizer chat template.
            prompts = []
            for sample in batch:
                system_prompt, user_prompt = build_prompt(self.task, sample)
                prompts.append(f"{system_prompt}\n\n{user_prompt}")

            outputs = self._safe_generate(prompts, sampling_params)

            for sample, output in zip(batch, outputs):
                if output is None:
                    continue
                output_text = output.outputs[0].text
                record = _make_record(sample, output_text)
                append_output(out_path, record)
                results.append(record)
                total_completed += 1

                if total_completed % self.CHECKPOINT_EVERY == 0:
                    write_checkpoint(
                        self.run_id, self.model_id, self.task,
                        sample["id"], total_completed,
                    )

        return results

    def _safe_generate(self, prompts: list, sampling_params) -> list:
        """
        Wraps self.llm.generate() in a try/except.
        Uses generate() (not chat()) to avoid tokenizer chat-template issues
        for models that don't define one (e.g. Tamil-Mistral-7B).
        On failure: logs the error, returns a list of None values so the
        caller can skip failed prompts without aborting the entire run.
        """
        try:
            return self.llm.generate(prompts, sampling_params=sampling_params)
        except Exception as exc:
            print(f"[VLLMWorker] Batch inference failed ({len(prompts)} prompts): {exc}")
            return [None] * len(prompts)
