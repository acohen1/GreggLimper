from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import numpy as np, random, string
from gregg_limper.config import Config

def _rand_name(prefix="tmp"):
    return f"{prefix}_" + "".join(random.choice(string.ascii_lowercase) for _ in range(8))

def _gpu_index_candidates(metric="L2"):
    # Try several GPU-capable indexes; stop at first success.
    # Params chosen to be quick/minimal.
    return [
        ("GPU_CAGRA", {"metric_type": metric, "index_type": "GPU_CAGRA",
                       "params": {"graph_degree": 16, "itopk": 64, "search_width": 32}}),
        ("GPU_IVF_FLAT", {"metric_type": metric, "index_type": "GPU_IVF_FLAT",
                          "params": {"nlist": 256}}),
        ("GPU_IVF_PQ", {"metric_type": metric, "index_type": "GPU_IVF_PQ",
                        "params": {"nlist": 256, "m": 8, "nbits": 8}}),
        ("GPU_BRUTE_FORCE", {"metric_type": metric, "index_type": "GPU_BRUTE_FORCE",
                             "params": {}}),
    ]

def validate_connection() -> None:
    """Ensure Milvus is reachable and GPU-capable by creating a tiny GPU index."""
    connections.connect(alias="default", uri=f"http://{Config.MILVUS_HOST}:{Config.MILVUS_PORT}")
    _ = utility.get_server_version()  # connectivity check

    dim = 64
    name = _rand_name("gl_gpu_check")
    schema = CollectionSchema([
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ], description="gpu-capability-check")

    coll = Collection(name=name, schema=schema)

    xb = np.random.random((2000, dim)).astype("float32")
    coll.insert([xb])                         # auto_id PK
    coll.flush()

    last_err = None
    for label, idx in _gpu_index_candidates(metric="L2"):
        try:
            coll.create_index(field_name="vec", index_params=idx)
            # success -> cleanup and return
            coll.drop()
            return
        except Exception as e:
            last_err = (label, e)

    # none worked
    coll.drop()
    lab, e = last_err if last_err else ("<none attempted>", RuntimeError("unknown error"))
    raise RuntimeError(f"Milvus GPU capability check failed; last index '{lab}' error: {e}")
