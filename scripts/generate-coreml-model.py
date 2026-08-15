#!/usr/bin/env python3
"""Generate Apollo's deterministic temporal Core ML oracle equivalent."""

import argparse
import struct
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.models import datatypes
from coremltools.models.neural_network import NeuralNetworkBuilder


FEATURE_COUNT = 16
OUTPUT_NAMES = ("load", "transition", "pressure", "p95")
BIASES = np.array([0.06, 0.04, 0.05, 0.06], dtype=np.float32)
WEIGHTS = np.array(
    [
        [0.56, 0.16, 0.00, 0.00, 0.06, 0.02, 0.04, 0.00, 0.12, 0.02, 0.00, 0.00, 0.04, 0.06, -0.02, 0.02],
        [0.10, 0.12, 0.40, 0.18, 0.04, 0.06, 0.02, 0.02, 0.04, 0.00, 0.02, 0.00, 0.04, 0.02, -0.01, 0.06],
        [0.10, 0.06, 0.08, 0.04, 0.46, 0.18, 0.04, 0.02, 0.02, 0.16, 0.08, 0.10, 0.04, 0.04, 0.00, 0.02],
        [0.12, 0.04, 0.16, 0.08, 0.18, 0.08, 0.26, 0.14, 0.04, 0.06, 0.04, 0.04, 0.02, 0.02, 0.02, 0.04],
    ],
    dtype=np.float32,
)
SCHEMA_BYTES = (
    b"apollo.temporal.v1\0load\0load_delta\0transition\0transition_delta\0"
    b"pressure\0pressure_delta\0p95\0p95_delta\0cpu_utilization\0"
    b"memory_pressure\0io_pressure\0thermal_pressure\0run_queue\0active_work\0"
    b"sample_age\0load_transition_coupling\0"
)


def fnv1a64(data: bytes, initial: int = 0xCBF29CE484222325) -> int:
    value = initial
    for byte in data:
        value ^= byte
        value = (value * 0x00000100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value


def model_hash(schema_hash: int) -> int:
    value = fnv1a64(b"apollo.fixed-bounded-linear-model.v1\0")
    for output in range(len(OUTPUT_NAMES)):
        value = fnv1a64(struct.pack("<f", float(BIASES[output])), value)
        for weight in WEIGHTS[output]:
            value = fnv1a64(struct.pack("<f", float(weight)), value)
    return fnv1a64(struct.pack("<Q", schema_hash), value)


def generate(destination: Path) -> None:
    inputs = [("temporal_features", datatypes.Array(FEATURE_COUNT))]
    outputs = [(name, datatypes.Array(1)) for name in OUTPUT_NAMES]
    builder = NeuralNetworkBuilder(inputs, outputs, disable_rank5_shape_mapping=True)
    for index, name in enumerate(OUTPUT_NAMES):
        raw_name = f"{name}_raw"
        builder.add_inner_product(
            name=f"{name}_linear",
            W=WEIGHTS[index : index + 1],
            b=BIASES[index : index + 1],
            input_channels=FEATURE_COUNT,
            output_channels=1,
            has_bias=True,
            input_name="temporal_features",
            output_name=raw_name,
        )
        builder.add_clip(
            name=f"{name}_bounded",
            input_name=raw_name,
            output_name=name,
            min_value=0.0,
            max_value=1.0,
        )

    schema_hash = fnv1a64(SCHEMA_BYTES)
    oracle_hash = model_hash(schema_hash)
    spec = builder.spec
    spec.description.metadata.shortDescription = "Apollo temporal predictor v1"
    spec.description.metadata.userDefined["apollo_schema_hash"] = hex(schema_hash)
    spec.description.metadata.userDefined["apollo_model_hash"] = hex(oracle_hash)
    spec.description.metadata.userDefined["apollo_generator"] = "coremltools-9.0"
    destination.parent.mkdir(parents=True, exist_ok=True)
    ct.models.MLModel(spec).save(str(destination))
    print(f"generated={destination}")
    print(f"schema_hash=0x{schema_hash:016x}")
    print(f"model_hash=0x{oracle_hash:016x}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/apollo-temporal-v1.mlmodel"),
    )
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
