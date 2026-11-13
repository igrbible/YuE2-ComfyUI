# Ruach Studio's derivative of YuE2-ComfyUI

This branch, `vae-rs-x3`, is [Ruach Studio](https://github.com/igrbible/Ruach_Studio)'s derivative of [pytraveler/YuE2-ComfyUI](https://github.com/pytraveler/YuE2-ComfyUI). It starts from the upstream release 0.15.0 (`408862d`, tag `v0.15.0`), exactly as we took it, and adds one commit of ours on top.

## What it changes

- The `vae` choice gains `vae-rs-x3-v1` and `vae-rs-x3-v2`, Ruach Studio's grown decoders (YuE2-VAE-RS ×3) for the same latents as m-a-p's Standard decoder. The backbone is found as for `standard`; the decoder file rides beside it, and a run without the file is refused before anything is read.
- The decoder is read by Ruach Studio's `yue2_vae_rs.py`, vendored unchanged into `yue2_comfy/vendor/yue2_vae_rs/` with its NOTICE: float32 only, the ×3 layout and nothing else, every tensor present, the training step in the file's metadata telling v1 from v2, the 60-frame halo enforced.
- The file is found by its published name or its header, fetched into `models/vae` with its SHA-256 checked, and with `download` off the node names the link, the path and the hash. `IS_CHANGED` includes the decoder file's stamp.
- Tooltips in English and Russian, README and README_RU, NOTICE.md, `vendor/README.md`, and `tests/test_vae_rs.py` (19 tests).

## Why it stays here

We prepared this as a pull request for the upstream issue #16. Before we opened it, the author shipped his own integration of the same decoders in 0.16.0, so no pull request was opened: a second one would have taken his time instead of saving it. We ran the same checks on his 0.16.0 and gave him the numbers in #16. This branch is kept as the record of our work, for anyone who wants to compare or take something from it. The fork's `main` follows the upstream.

## HERESY numbers

Our numbered patch log, `docs/HERESY.md` in Ruach Studio, has no entry of its own for this branch: it was made outside the Studio's tree. The decoders and the loader it vendors are HERESY 2304 (the ×3 decoders in the Studio) and 2329 (the decoders finished for 2.0.7, the integration kit released under Apache-2.0).

## Licence

The changes in this branch are under the Apache License 2.0, the licence of the upstream project, so they can be taken back into it as they are. The vendored `yue2_vae_rs.py` is Apache-2.0 as Ruach Studio released it with the decoders, and its NOTICE (crediting "Ruach Studio and Viktor Zhuromskyy") goes with it. Ruach Studio itself is AGPL-3.0-or-later; that licence covers the Studio, not this branch. The decoder weights are not covered by any of this: YuE2-VAE-RS-x3_v1 and _v2 are adaptations of m-a-p's YuE2-Vae and are CC BY-NC 4.0, non-commercial, crediting YuE2 and m-a-p, and Ruach Studio for VAE-RS.

## Openly

This is a derivative and we say so. We hide nothing and are not ashamed of it: what we took, from whom, at which commit, and what we added are all written here and in the commit.
