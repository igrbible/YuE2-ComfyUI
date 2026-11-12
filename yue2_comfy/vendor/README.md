# Vendored upstream code -- do not edit

`yue2/` is a verbatim copy of `src/yue2/` from
https://github.com/multimodal-art-projection/YuE (Apache-2.0), version 0.1.6.

Keeping it unmodified is the point: updating means copying the new files over,
and every diff stays about what upstream changed.

## What is here

    protocol.py          prompting, sampling defaults, token layout
    modeling_yue2.py     the AR/NAR mixture-of-transformers backbone
    modeling_vae.py      the Oobleck VAE that turns latents into audio
    tokenization_yue2.py the text and score BPE
    sampling.py          the autoregressive decode loop
    nar.py               acoustic flow matching
    cuda_graph.py        pure-PyTorch CUDA graph decode

## What is not, and why

    cli.py          command line entry point; not used, and its example lyrics
                    are the only other file with non-ASCII content
    fast.py         the vLLM backend; fcntl, os.killpg and SIGKILL are POSIX
                    only and this pack does not offer that backend
    pipeline.py     YuE2Pipeline; it sets six torch globals process-wide, which
                    inside ComfyUI would degrade every other model until
                    restart. This pack orchestrates the four stages itself and
                    scopes those flags.
    storage.py      HF resolution and artifact hashing; this pack does its own
    progress.py     stderr progress; replaced by the ComfyUI node progress
    quantization.py opt-in FP8; not offered

`tokenization_yue2.py` imports `storage` inside `from_pretrained`, which this
pack never calls.

## yue2_music/

`yue2_music/abc_tools.py` is a verbatim copy of
`skills/yue2-music/scripts/abc_tools.py` from the same repository and version,
which ships the skill with its own copy of the same Apache-2.0 licence. It is
upstream's reader for the two-voice ABC dialect the model writes, and
`yue2_comfy/transpose.py` leans on it twice: to read a score before moving it,
and to read the moved score back and compare it with the original note by note.
Taking upstream's reader rather than writing a second one leaves the dialect's
conventions -- an accidental carrying to every octave of its letter until the
barline, among them -- for upstream to get right. Its command line entry point
is never called.

`yue2_music/__init__.py` is this pack's own and only makes the folder a package.

## The ASCII rule does not apply here

This project's own source is ASCII only. `modeling_yue2.py` contains 833
characters above U+007E -- section banners drawn with U+2500 and U+2550, a few
dashes and arrows, all in comments -- and `abc_tools.py` one, an en dash in an
error message. Scrubbing them would mean maintaining a fork and making every
future upstream diff noisy, for no runtime benefit. The ASCII check skips this
directory.

## yue2_vae_rs/

`yue2_vae_rs/yue2_vae_rs.py` is a verbatim copy of `VAE-RS/kit/yue2_vae_rs.py`
(version 1.0.0, SHA-256
`b7fd8baeddbe1f8c5dd9940cc071d9e4dd4a0f137ef695cb8f301396ecfb21a1`) from
https://huggingface.co/goldhub/YuE2-RS-Finetunes, Ruach Studio's loader and
decoder for its YuE2-VAE-RS x3 decoders (Apache-2.0). `NOTICE` beside it is
that kit's notice, which section 4 (d) of the licence asks to travel with the
code. It needs nothing but torch and safetensors.

The pack reads the x3 decoders through it and nothing else, for two reasons.
The file layout is its own (`conv1`, `pre`, `blocks`, ...), which the vendored
`modeling_vae.py` cannot hold: three new residual units at the latent rate and
three more in each upsampling block. And its `load_decoder` is strict in every
way the pack's own loader is: float32 tensors only, the x3 shape and no other,
every tensor present and none left over, the file's metadata checked against
the shapes. Its decoder object speaks the decode API of `modeling_vae.py`
(`decode`, `decode_tiled`, `required_halo`, `natural_output_length`), so the
decode stage runs it unchanged.

`yue2_vae_rs/__init__.py` is this pack's own and only makes the folder a
package.
