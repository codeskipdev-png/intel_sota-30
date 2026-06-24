"""Reference Poker44 miner with simple chunk-level behavioral heuristics."""

# from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path
from typing import Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from detect_bots import detect_bots
from poker44.validator.synapse import DetectionSynapse
import json
from typing import List, Any


class Miner(BaseMinerNeuron):
    """
    Reference heuristic miner.

    It aggregates simple behavior signals over each chunk and returns a bot-risk
    score per chunk. The goal is not SOTA accuracy, but a deterministic and
    explainable baseline that is meaningfully better than random.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        bt.logging.info("🤖 Heuristic Poker44 Miner started")
        repo_root = Path(__file__).resolve().parents[1]
        self.output_dir = repo_root / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=[Path(__file__).resolve()],
            defaults={
                "model_name": "poker44-sota-model",
                "model_version": "1.1",
                "framework": "python-heuristic",
                "license": "MIT",
                "repo_url": "https://github.com/codeskipdev-png/intel_sota",
                "repo_commit": "060c97a",
                "notes": "Reference heuristic miner shipped with the Poker44 subnet.",
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Reference heuristic miner. No training step. Uses only runtime chunk features."
                ),
                "training_data_sources": ["none"],
                "private_data_attestation": (
                    "This reference miner does not train on validator-only evaluation data."
                ),
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup(repo_root)
        self.reqeust_count = 0
        
        # # Attach handlers after initialization
        # self.axon.attach(
        #     forward_fn = self.forward,
        #     blacklist_fn = self.blacklist,
        #     priority_fn = self.priority,
        # )
        # bt.logging.info("Attaching forward function to miner axon.")
        
        bt.logging.info(f"Axon created: {self.axon}")

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info("Open-sourced miner manifest standard active for this miner.")
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"open_source={self.model_manifest.get('open_source')}"
        )
        bt.logging.info(
            f"Manifest digest={self.manifest_digest} "
            f"inference_mode={self.model_manifest.get('inference_mode', '')}"
        )
        bt.logging.info(
            "Miner prep docs available | "
            f"miner_doc={repo_root / 'docs' / 'miner.md'}"
        )

    def save_chunks(self, chunks: List[List[dict[str, Any]]]):
        output_file = self.output_dir / f"chunks_{self.reqeust_count}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        """Assign one deterministic bot-risk score per chunk."""
        chunks = synapse.chunks or []
        self.reqeust_count += 1
        print(f"Request count: {self.reqeust_count}")

        # try:
        #     self.save_chunks(chunks)
        # except Exception as e:
        #    print(f"Error saving chunks: {e}")
        

        start_time = time.time()
        risk_scores, predictions = detect_bots(chunks)
        end_time = time.time()

        elapsed_time = end_time - start_time
        bt.logging.info(f"Time taken: {elapsed_time:.4f} seconds")
        print(f"Elapsed time: {elapsed_time:.4f} seconds")


        synapse.risk_scores = risk_scores
        synapse.predictions = predictions
        synapse.model_manifest = dict(self.model_manifest)
        bt.logging.info(f"Miner Predctions: {synapse.predictions}")
        bt.logging.info(f"Scored {len(chunks)} chunks with heuristic risks.")
        print(f"Miner Predctions: {synapse.predictions}")
        print(f"Scored {len(chunks)} chunks with heuristic risks.")
        return synapse


    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        """Determine whether to blacklist incoming requests."""
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        """Assign priority based on caller's stake."""
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:        
        bt.logging.info("sota-heuristic miner running...")
        while True:
            print(f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}")
            bt.logging.info(f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}")
            time.sleep(5 * 60)
