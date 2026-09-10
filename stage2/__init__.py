"""
Stage 2 of the multistage mosquito-larvae framework: genus-level classification
with InceptionV3, run on Stage2_new_aug_dataset.

Modules
-------
dedup       near-duplicate grouping (defines what "one source photograph" means)
split       leakage-free, group-aware, class-stratified train/val/test manifests
data        manifest-backed Dataset + train/eval transform pipelines
model       InceptionV3 builder with dropout and layer-freezing control
metrics     accuracy / precision / recall / F1 / confusion / AUC-ROC / CIs
engine      training loop (AMP, schedulers, early stopping, timing) and evaluation
experiments the experiment registry: one baseline plus the tuning grid
run         CLI entry point
report      comparison tables, curves and confusion-matrix figures
"""
