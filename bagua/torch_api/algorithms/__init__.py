#!/usr/bin/env python3

from .base import Algorithm, AlgorithmImpl, GlobalAlgorithmRegistry  # noqa: F401
from . import bytegrad, decentralized, gradient_allreduce  # noqa: F401
from . import q_adam, async_model_average, sketch  # noqa: F401
from . import qsparselocal


GlobalAlgorithmRegistry.register(
    "gradient_allreduce",
    gradient_allreduce.GradientAllReduceAlgorithm,
    description="Gradient AllReduce Algorithm",
)
GlobalAlgorithmRegistry.register(
    "bytegrad", bytegrad.ByteGradAlgorithm, description="ByteGrad Algorithm"
)
GlobalAlgorithmRegistry.register(
    "decentralized",
    decentralized.DecentralizedAlgorithm,
    description="Decentralized SGD Algorithm",
)
GlobalAlgorithmRegistry.register(
    "low_precision_decentralized",
    decentralized.LowPrecisionDecentralizedAlgorithm,
    description="Low Precision Decentralized SGD Algorithm",
)
GlobalAlgorithmRegistry.register(
    "qadam", q_adam.QAdamAlgorithm, description="QAdam Algorithm"
)
GlobalAlgorithmRegistry.register(
    "async",
    async_model_average.AsyncModelAverageAlgorithm,
    description="Asynchronous Model Average Algorithm",
)
GlobalAlgorithmRegistry.register(
    "qgadam", q_adam.QGAdamAlgorithm, description="QGAdam Algorithm"
)
GlobalAlgorithmRegistry.register(
    "floatgrad", bytegrad.Float16GradAlgorithm, description="Float16 Algorithm"
)
GlobalAlgorithmRegistry.register(
    "qgadam_low_precision_decentralized", decentralized.QGAdamLowPrecisionDecentralizedAlgorithm, description="QGAdam Low Precision Decentralized Algorithm"
)
GlobalAlgorithmRegistry.register(
    "gradient_allreduce_sketch", gradient_allreduce.GradientAllReduceSketchAlgorithm, description="Gradient AllReduce Sketch Algorithm"
)
GlobalAlgorithmRegistry.register(
    "sketch", sketch.SketchAlgorithm, description="Sketch Algorithm"
)
GlobalAlgorithmRegistry.register(
    "qsparse", qsparselocal.QSparseLocalAlgorithm, description="Qsparselocal Algorithm"
)