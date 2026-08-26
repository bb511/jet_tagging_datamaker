# The loading pipeline that ships with the record.
#
# Three stages run in order and the last one caches what it wrote, so a rerun over the
# same cache directory costs nothing:
#
#   reading      the published parquet shards -> the padded (jets, constituents,
#                features) array the study's own pipeline works on
#   physics      keep the leading constituents by pT, fit the scale-only normalisation
#                on the training split, apply it
#   datamodule   drive both for a caller who wants tensors and nothing else
#
# The stages are configured by the hydra tree under configs/, which names them by their
# _target_. datamodule.JetData is the only entry point a consumer needs.
#
# Nothing is imported here, so that reading and physics can be used without torch
# installed.
