# Model artifacts

The pipeline writes model artifacts to this directory. Generate them with the
numbered runners described in the [README](../README.md).

```text
DIM/<model_id>/
    direction.pt
    direction_metadata.json
caa_vectors_raw/<behavior>/
    vec_layer_<layer>_<model>.pt
caa_vectors/<behavior>/
    vec_layer_<layer>_<model>.pt
cast_vectors/<model_id>/<variant>/
    final_vector_<behavior>_<model_id>_<loss>.pt
    sanitized_<behavior>_<model_id>_<loss>.pt
```

`final_vector_*.pt` contains the learned CAST basis. `sanitized_*.pt` contains
the vector used for inference. The default variant is `dim_1_masked_kl`.
Generated artifacts are gitignored.
