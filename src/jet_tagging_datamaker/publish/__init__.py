# Packaging the Zenodo release as a HuggingFace record.
#
# export      the Zenodo archives, the h5 files behind them, and the tables they become
# huggingface the row-per-jet mirror: shards, card, assets
# card        the dataset card and the licence that ship inside the mirror
# assets      the loading pipeline and its configs, which ship inside the mirror rather
#             than run from here
