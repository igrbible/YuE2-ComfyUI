"""Ruach Studio's YuE2-VAE-RS x3 decoders as a 'vae' choice, checked without torch or the weights.

What is checked here is everything around the decoder: that the search finds
the right release and never mistakes one for another or for m-a-p's decoder,
that a run asking for one gets it or a refusal, never Standard's decoder under
its name, and that the cache tells the decoders apart. That the 427 real
tensors load and decode as published is checked by running it against the
reference that ships with the decoders, not by a unit test.
"""

import json
import os
import struct

import pytest

from yue2_comfy import constants, discovery, download, loader

V1, V2 = "vae-rs-x3-v1", "vae-rs-x3-v2"


def write_x3(path, step, extra=(), variant="x3", markers=constants.VAE_RS_MARKERS):
    """A header-only stand-in for an x3 file: its marker tensors, its metadata, zero-filled scalars."""
    header = {}
    offset = 0
    for name in tuple(markers) + tuple(extra):
        header[name] = {"dtype": "F32", "shape": [1], "data_offsets": [offset, offset + 4]}
        offset += 4
    header["__metadata__"] = {"format": "pt", "variant": variant, "step": str(step)}
    body = json.dumps(header).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(body)))
        handle.write(body)
        handle.write(b"\0" * offset)
    return str(path)


def test_the_new_choices_come_after_the_old_ones():
    """A saved workflow keeps its 'vae' value, and the default stays 'standard'."""
    assert constants.VAE_CHOICES[:2] == ("standard", "legacy")
    assert set(constants.VAE_CHOICES[2:]) == set(constants.VAE_RS_RELEASES)
    assert constants.DEFAULT_OPTIONS["vae"] == "standard"


def test_the_two_releases_differ_only_in_what_tells_them_apart():
    v1, v2 = constants.VAE_RS_RELEASES[V1], constants.VAE_RS_RELEASES[V2]
    assert v1["bytes"] == v2["bytes"]
    assert v1["step"] != v2["step"] and v1["sha256"] != v2["sha256"]
    for release in (v1, v2):
        assert release["path"].startswith("VAE-RS/") and release["path"].endswith(".safetensors")
        assert len(release["sha256"]) == 64


@pytest.mark.parametrize("step, expected", [("161953", V1), ("155030", V2), ("291130", ""), ("", "")])
def test_a_release_is_known_by_its_step(tmp_path, step, expected):
    assert discovery.vae_rs_release(write_x3(tmp_path / "x.safetensors", step)) == expected


def test_another_variant_is_not_taken_for_x3(tmp_path):
    assert discovery.vae_rs_release(write_x3(tmp_path / "x4.safetensors", "161953", variant="x4")) == ""


def test_m_a_p_s_decoder_is_not_taken_for_x3_nor_x3_for_it(tmp_path):
    """The two layouts share no marker, so neither search stops on the other's file."""
    standard = tmp_path / "standard.safetensors"
    write_x3(standard, "161953", markers=constants.VAE_MARKERS)
    assert discovery.vae_rs_release(str(standard)) == ""
    x3 = write_x3(tmp_path / "x3.safetensors", "161953")
    assert discovery.identify(x3) != "vae"


def test_the_published_name_is_found_and_confirmed_by_its_header(tmp_path):
    """A v2 file under v1's name is not v1, whatever it is called."""
    vae = tmp_path / "vae"
    vae.mkdir()
    write_x3(vae / "YuE2-VAE-RS-x3_v1.safetensors", "155030")
    assert discovery.find_vae_rs(V1, [str(tmp_path)]) == ""
    assert discovery.find_vae_rs(V2, [str(tmp_path)]) == str(vae / "YuE2-VAE-RS-x3_v1.safetensors")


def test_the_decoder_only_file_wins_over_the_full_one(tmp_path):
    full = write_x3(tmp_path / constants.VAE_RS_RELEASES[V1]["full"], "161953", extra=("encoder.layers.0.bias",))
    assert discovery.find_vae_rs(V1, [str(tmp_path)]) == full
    only = write_x3(tmp_path / "YuE2-VAE-RS-x3_v1.safetensors", "161953")
    assert discovery.find_vae_rs(V1, [str(tmp_path)]) == only


def test_a_renamed_file_is_found_by_its_header(tmp_path):
    renamed = write_x3(tmp_path / "my decoder.safetensors", "155030")
    assert discovery.find_vae_rs(V2, [str(tmp_path)]) == renamed
    assert discovery.find_vae_rs(V1, [str(tmp_path)]) == ""
    assert discovery.find_vae_rs("standard", [str(tmp_path)]) == ""


def test_a_repack_alone_is_refused_for_x3_before_anything_is_read(tmp_path):
    """Otherwise the repack's own decoder would be built and called x3."""
    missing = str(tmp_path / "not-there.safetensors")
    with pytest.raises(ValueError, match="separate file"):
        loader.load_repack(missing, "cpu", V1)


def test_the_cache_tells_the_decoders_apart(tmp_path):
    files = loader.Files(repack=write_x3(tmp_path / "r.safetensors", "0"),
                         vae=write_x3(tmp_path / "v.safetensors", "161953"))
    keys = {loader._cache_key(files, "cpu", variant) for variant in constants.VAE_CHOICES}
    assert len(keys) == len(constants.VAE_CHOICES)


def test_an_x3_run_finds_its_backbone_as_standard_and_its_decoder_beside_it(tmp_path, monkeypatch):
    seen = {}

    def backbone(settings, progress=None):
        seen.update(settings)
        return discovery.Files(repack="repack.safetensors")

    decoder = write_x3(tmp_path / "YuE2-VAE-RS-x3_v2.safetensors", "155030")
    monkeypatch.setattr(download, "_ensure", backbone)
    monkeypatch.setattr(discovery, "find_vae_rs", lambda variant, roots=None: decoder if variant == V2 else "")
    files = download.ensure({"vae": V2, "download": "off"})
    assert seen["vae"] == "standard"
    assert files.repack == "repack.safetensors" and files.vae == decoder


def test_other_runs_are_handed_through_unchanged(monkeypatch):
    seen = {}
    monkeypatch.setattr(download, "_ensure", lambda settings, progress=None: seen.update(settings) or "files")
    for variant in ("standard", "legacy"):
        assert download.ensure({"vae": variant}) == "files"
        assert seen["vae"] == variant


def test_with_downloads_off_the_refusal_says_where_the_file_is_and_goes(tmp_path, monkeypatch):
    monkeypatch.setenv("YUE2_MODELS_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError) as refused:
        download.ensure_vae_rs(V1, {"download": "off"})
    text = str(refused.value)
    release = constants.VAE_RS_RELEASES[V1]
    assert "https://huggingface.co/" + constants.VAE_RS_REPO + "/resolve/main/" + release["path"] in text
    assert os.path.join(str(tmp_path), "vae", "YuE2-VAE-RS-x3_v1.safetensors") in text
    assert release["sha256"] in text and release["full"] in text


def test_a_download_that_does_not_match_its_hash_is_removed(tmp_path, monkeypatch):
    monkeypatch.setenv("YUE2_MODELS_ROOT", str(tmp_path))

    def fake_fetch(repo, wanted, title, progress=None):
        for dest in wanted.values():
            with open(dest, "wb") as handle:
                handle.write(b"not the decoder")
        return 15

    monkeypatch.setattr(download, "fetch", fake_fetch)
    with pytest.raises(download.DownloadError, match="SHA-256"):
        download.ensure_vae_rs(V1, {"download": "auto"})
    assert not os.path.exists(os.path.join(str(tmp_path), "vae", "YuE2-VAE-RS-x3_v1.safetensors"))


def test_the_options_node_answers_as_before_for_the_old_choices():
    """Only an x3 choice adds its decoder's identity to the cache answer."""
    from yue2_comfy.nodes import YuE2Options

    assert YuE2Options.IS_CHANGED(checkpoint="auto", vae="standard") == "auto"
    assert YuE2Options.IS_CHANGED(checkpoint="auto", vae="legacy") == "auto"


def test_a_replaced_decoder_changes_the_options_cache_answer(tmp_path, monkeypatch):
    from yue2_comfy.nodes import YuE2Options

    path = write_x3(tmp_path / "YuE2-VAE-RS-x3_v1.safetensors", "161953")
    monkeypatch.setattr(discovery, "find_vae_rs", lambda variant, roots=None: path)
    first = YuE2Options.IS_CHANGED(checkpoint="auto", vae=V1)
    assert first != "auto" and first == YuE2Options.IS_CHANGED(checkpoint="auto", vae=V1)
    write_x3(path, "161953", extra=("x",))
    assert YuE2Options.IS_CHANGED(checkpoint="auto", vae=V1) != first
