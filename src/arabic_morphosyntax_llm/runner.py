"""Provider-neutral LLM benchmark execution."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict
from typing import Any

from .config import ModelProfile
from .prompts import build_request
from .providers.base import Provider
from .records import GenerationRequest, GenerationResult
from .retrieval import Selection, select_demonstrations
from .validation import validate_prediction


class PipelineError(RuntimeError):
    pass


def request_as_dict(request: GenerationRequest) -> dict[str, Any]:
    return {
        "system_prompt": request.system_prompt,
        "messages": [asdict(message) for message in request.messages],
        "schema_name": request.schema_name,
        "schema": request.schema,
    }


class TaskPipeline:
    def __init__(
        self,
        *,
        provider: Provider,
        profile: ModelProfile,
        task: str,
        demonstrations: list[dict[str, Any]],
        shots: int,
        retrieval_method: str,
        seed: int,
        retries: int,
        shot_map: list[dict[str, Any]] | None = None,
    ):
        self.provider = provider
        self.profile = profile
        self.task = task
        self.demonstrations = demonstrations
        self.shots = shots
        self.retrieval_method = retrieval_method
        self.seed = seed
        self.retries = retries
        self.shot_map = {row["id"]: row["shots"] for row in shot_map} if shot_map else None
        self.demo_by_id = {row["id"]: row for row in demonstrations}
        if self.shot_map is not None:
            if len(self.demo_by_id) != len(demonstrations):
                raise ValueError("demonstration IDs must be unique")
            for query_id, entries in self.shot_map.items():
                expected = min(shots, len(demonstrations))
                if len(entries) != expected:
                    raise ValueError(f"shot map {query_id}: expected {expected} shots")
                for entry in entries:
                    if entry["id"] not in self.demo_by_id:
                        raise ValueError(f"shot map {query_id}: unknown demonstration {entry['id']}")

    def select(self, record: dict[str, Any]) -> Selection:
        if self.shot_map is not None:
            if record["id"] not in self.shot_map:
                raise ValueError(f"shot map is missing query {record['id']}")
            entries = self.shot_map[record["id"]]
            return Selection([self.demo_by_id[entry["id"]] for entry in entries], entries)
        return select_demonstrations(
            record,
            self.demonstrations,
            task=self.task,
            shots=self.shots,
            method=self.retrieval_method,
            seed=self.seed,
        )

    async def _generate(
        self,
        *,
        stage: str,
        prompt_task: str,
        request: GenerationRequest,
        record: dict[str, Any],
    ) -> tuple[GenerationResult, dict[str, Any]]:
        failures: list[str] = []
        for attempt in range(1, self.retries + 2):
            try:
                result = await self.provider.generate(request)
                validate_prediction(prompt_task, result.data, record)
                metadata = result.metadata(stage)
                metadata["attempt"] = attempt
                serialized_request = json.dumps(
                    request_as_dict(request),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                metadata["request_sha256"] = hashlib.sha256(serialized_request).hexdigest()
                metadata["system_prompt_sha256"] = hashlib.sha256(
                    request.system_prompt.encode("utf-8")
                ).hexdigest()
                metadata["schema_sha256"] = hashlib.sha256(
                    json.dumps(
                        request.schema, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest()
                if failures:
                    metadata["previous_failures"] = failures
                return result, metadata
            except Exception as error:  # provider and validation failures are checkpointed upstream
                failures.append(f"{type(error).__name__}: {error}")
                if attempt > self.retries:
                    break
                await asyncio.sleep(min(2 ** (attempt - 1), 8))
        raise PipelineError(
            f"{stage} failed after {self.retries + 1} attempt(s): {failures[-1]}"
        )

    async def predict(self, record: dict[str, Any]) -> dict[str, Any]:
        selection = self.select(record)
        calls: list[dict[str, Any]] = []

        if self.task == "cascade":
            morph_request = build_request("morphology", record, selection.records)
            morph_result, morph_metadata = await self._generate(
                stage="morphology",
                prompt_task="morphology",
                request=morph_request,
                record=record,
            )
            calls.append(morph_metadata)
            predicted_morphology = morph_result.data["tokens"]
            parse_request = build_request(
                "cascade-parse",
                record,
                selection.records,
                predicted_morphology=predicted_morphology,
            )
            parse_result, parse_metadata = await self._generate(
                stage="morphology_guided_parsing",
                prompt_task="cascade-parse",
                request=parse_request,
                record=record,
            )
            calls.append(parse_metadata)
            prediction: dict[str, Any] = {
                "morphology": predicted_morphology,
                "parses": parse_result.data["parses"],
            }
        else:
            request = build_request(self.task, record, selection.records)
            result, metadata = await self._generate(
                stage=self.task,
                prompt_task=self.task,
                request=request,
                record=record,
            )
            calls.append(metadata)
            prediction = result.data

        input_snapshot = {"id": record["id"]}
        for key in ("text", "tokens"):
            if key in record:
                input_snapshot[key] = record[key]
        return {
            "id": record["id"],
            "task": self.task,
            "model": self.profile.key,
            "model_id": self.profile.model_id,
            "configuration": {
                "provider": self.profile.provider,
                "base_url": self.profile.base_url,
                "generation": asdict(self.profile.generation),
                "provider_options": self.profile.provider_options,
            },
            "status": "ok",
            "input": input_snapshot,
            "prediction": prediction,
            "demonstrations": selection.metadata,
            "calls": calls,
            "error": None,
        }


def render_requests(
    *,
    task: str,
    record: dict[str, Any],
    selection: Selection,
) -> list[dict[str, Any]]:
    if task != "cascade":
        return [{"stage": task, **request_as_dict(build_request(task, record, selection.records))}]

    rendered = [
        {
            "stage": "morphology",
            **request_as_dict(build_request("morphology", record, selection.records)),
        }
    ]
    supplied = record.get("morphology")
    if supplied is not None:
        rendered.append(
            {
                "stage": "morphology_guided_parsing",
                **request_as_dict(
                    build_request(
                        "cascade-parse",
                        record,
                        selection.records,
                        predicted_morphology=supplied,
                    )
                ),
            }
        )
    else:
        rendered.append(
            {
                "stage": "morphology_guided_parsing",
                "deferred": True,
                "reason": "The parser request is rendered after stage-one morphology is predicted.",
            }
        )
    return rendered
