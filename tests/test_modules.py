import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from safetensors.numpy import load_file

from app.imaging import ssim_images
from app.modules import epoch_picker, hash_index, merge_recipe, node_profiler, provenance, seed_variance, shelf_life, style_bleed
from app.paths import SAMPLE_DIR

REPO = Path(__file__).resolve().parents[1]


def test_ssim_identity_and_difference():
    flat = Image.new("RGB", (32, 32), (20, 30, 40))
    assert ssim_images(flat, flat) == pytest.approx(1.0)
    other = Image.new("RGB", (32, 32), (220, 10, 10))
    assert ssim_images(flat, other) < ssim_images(flat, flat)
    black = Image.new("RGB", (32, 32), (0, 0, 0))
    white = Image.new("RGB", (32, 32), (255, 255, 255))
    assert ssim_images(black, white) < 0.05


def test_grid_and_queue_match_sample_images(tmp_path):
    grid = epoch_picker.build_grid(SAMPLE_DIR)
    assert grid["epochs"] == ["epoch_01", "epoch_02", "epoch_03", "epoch_04"]
    assert [prompt["id"] for prompt in grid["prompts"]] == ["amber_circle", "teal_bars", "violet_triangle"]
    assert grid["seeds"] == [11, 22, 33]
    assert grid["image_count"] == 36
    written = epoch_picker.write_queue_file(SAMPLE_DIR, tmp_path, grid)
    assert written["submitted"] is False
    assert written["comfyui_called"] is False
    assert written["count"] == 36
    payload = json.loads((tmp_path / "queue" / "epoch_queue.json").read_text(encoding="utf-8"))
    assert payload["submitted"] is False
    seeds = {item["seed"] for item in payload["items"]}
    assert seeds == {11, 22, 33}
    sample = payload["items"][0]["workflow"]["6"]["inputs"]
    assert sample["sampler_name"] == "euler"
    assert "class_type" in payload["items"][0]["workflow"]["6"]


def test_picks_round_trip(tmp_path):
    saved = epoch_picker.save_pick(
        tmp_path,
        {"epoch": "epoch_02", "prompt_id": "amber_circle", "seed": 11, "path": "epochs/epoch_02/amber_circle/seed_11.png", "notes": "cleaner edge"},
    )
    assert saved["saved"]["notes"] == "cleaner edge"
    again = epoch_picker.save_pick(
        tmp_path,
        {"epoch": "epoch_02", "prompt_id": "amber_circle", "seed": 11, "path": "epochs/epoch_02/amber_circle/seed_11.png", "notes": "still the one"},
    )
    assert len(again["picks"]) == 1
    assert again["picks"][0]["notes"] == "still the one"


def test_sample_variance_flags_only_the_chaotic_epoch():
    report = seed_variance.variance_report(SAMPLE_DIR, method="ssim")
    assert report["method"] == "ssim"
    by_epoch = {row["epoch"]: row for row in report["epochs"]}
    assert by_epoch["epoch_03"]["unstable"] is True
    for name in ("epoch_01", "epoch_02", "epoch_04"):
        assert by_epoch[name]["unstable"] is False
        assert by_epoch[name]["mean"] > by_epoch["epoch_03"]["mean"]


def test_hand_built_variance_math(tmp_path):
    root = tmp_path / "sample"
    stable = root / "epochs" / "epoch_01" / "flat"
    wild = root / "epochs" / "epoch_02" / "flat"
    stable.mkdir(parents=True)
    wild.mkdir(parents=True)
    Image.new("RGB", (24, 24), (30, 30, 30)).save(stable / "seed_1.png")
    Image.new("RGB", (24, 24), (32, 32, 32)).save(stable / "seed_2.png")
    Image.new("RGB", (24, 24), (10, 10, 10)).save(wild / "seed_1.png")
    Image.new("RGB", (24, 24), (240, 240, 240)).save(wild / "seed_2.png")
    report = seed_variance.variance_report(root, method="ssim")
    flagged = set(report["unstable_epochs"])
    assert flagged == {"epoch_02"}
    assert report["method"] == "ssim"


def test_sample_bleed_flags_the_late_epoch():
    report = style_bleed.bleed_report(SAMPLE_DIR, method="ssim")
    assert report["method"] == "ssim"
    by_epoch = {row["epoch"]: row for row in report["epochs"]}
    assert by_epoch["epoch_01"]["flagged"] is False
    assert by_epoch["epoch_04"]["flagged"] is True
    assert by_epoch["epoch_04"]["leakage"] > by_epoch["epoch_01"]["leakage"]


def test_hand_built_bleed_math(tmp_path):
    root = tmp_path / "sample"
    low = root / "bleed" / "epoch_01" / "plain"
    high = root / "bleed" / "epoch_02" / "plain"
    for folder, color in ((low, (180, 180, 180)), (high, (20, 20, 220))):
        (folder / "without_lora").mkdir(parents=True)
        (folder / "with_lora").mkdir(parents=True)
        Image.new("RGB", (24, 24), (180, 180, 180)).save(folder / "without_lora" / "seed_1.png")
        Image.new("RGB", (24, 24), color).save(folder / "with_lora" / "seed_1.png")
    report = style_bleed.bleed_report(root, method="ssim", leakage_ceiling=0.2)
    by_epoch = {row["epoch"]: row for row in report["epochs"]}
    assert by_epoch["epoch_01"]["flagged"] is False
    assert by_epoch["epoch_02"]["flagged"] is True


def _reference_autov2(path: Path) -> str:
    data = path.read_bytes()
    header_len = struct.unpack_from("<Q", data, 0)[0]
    payload = data[8 + header_len :]
    return hashlib.sha256(payload).hexdigest()[:10]


def test_hash_and_metadata_on_sample_loras(tmp_path):
    folder = SAMPLE_DIR / "loras"
    rows = hash_index.scan_folder(folder, tmp_path / "catalog.sqlite")
    by_name = {row["filename"]: row for row in rows}
    assert set(by_name) == {
        "style_amber.safetensors",
        "style_teal.safetensors",
        "style_amber_copy.safetensors",
        "style_amber_retag.safetensors",
        "mystery_lora.safetensors",
        "old_unused.safetensors",
    }
    amber = by_name["style_amber.safetensors"]
    assert amber["sha256"] == hashlib.sha256((folder / "style_amber.safetensors").read_bytes()).hexdigest()
    assert amber["autov2"] == _reference_autov2(folder / "style_amber.safetensors")
    assert amber["base_model"] == "sdxl_base_v1-0"
    assert "amberstyle" in amber["trigger_words"]
    assert amber["metadata"]["ss_epoch"] == "4"
    assert amber["missing_metadata"] is False
    copy = by_name["style_amber_copy.safetensors"]
    retag = by_name["style_amber_retag.safetensors"]
    assert copy["sha256"] == amber["sha256"]
    assert copy["autov2"] == amber["autov2"]
    assert retag["autov2"] == amber["autov2"]
    assert retag["sha256"] != amber["sha256"]
    assert by_name["mystery_lora.safetensors"]["missing_metadata"] is True
    assert by_name["mystery_lora.safetensors"]["base_model"] == ""
    before = {path.name: path.stat().st_mtime_ns for path in folder.glob("*.safetensors")}
    hash_index.scan_folder(folder, tmp_path / "catalog.sqlite")
    after = {path.name: path.stat().st_mtime_ns for path in folder.glob("*.safetensors")}
    assert before == after


def test_png_and_history_provenance():
    record = provenance.parse_png(SAMPLE_DIR / "epochs" / "epoch_01" / "amber_circle" / "seed_11.png", SAMPLE_DIR)
    assert record["seed"] == 11
    assert record["sampler"] == "euler"
    assert record["loras"][0]["name"] == "style_amber.safetensors"
    assert record["loras"][0]["strength_model"] == pytest.approx(0.8)
    assert "amber circle" in record["prompt_text"]
    history = json.loads((SAMPLE_DIR / "history" / "history.json").read_text(encoding="utf-8"))
    rows = provenance.parse_history(history)
    assert {row["comfy_prompt_id"] for row in rows} == {"synthetic-amber-recent", "synthetic-teal-old"}
    usage = provenance.usage_by_lora(rows)
    assert usage["style_amber.safetensors"] == "2026-10-28"
    assert usage["style_teal.safetensors"] == "2026-08-01"
    found = provenance.search_records(rows, "style_teal")
    assert len(found) == 1


def test_node_ranking_from_sample_and_messages():
    history = json.loads((SAMPLE_DIR / "history" / "history.json").read_text(encoding="utf-8"))
    report = node_profiler.rank_nodes(history)
    assert report["read_only"] is True
    assert report["deleted"] == []
    assert report["comfyui_called"] is False
    assert report["slowest_type"] == "KSampler"
    assert report["slowest_node"]["node_id"] == "6"
    assert report["by_type"][0]["total_ms"] == pytest.approx(4200)
    messages = {
        "p1": {
            "prompt": [1, "p1", {
                "6": {"class_type": "KSampler", "inputs": {}},
                "8": {"class_type": "VAEDecode", "inputs": {}},
            }, {}, ["9"]],
            "status": {"messages": [
                ["execution_start", {"timestamp": 0}],
                ["executing", {"node": "6", "timestamp": 100}],
                ["executing", {"node": "8", "timestamp": 2100}],
                ["execution_success", {"timestamp": 2300}],
            ]},
        }
    }
    ranked = node_profiler.rank_nodes(messages)
    assert ranked["slowest_type"] == "KSampler"
    assert ranked["by_node"][0]["node_id"] == "6"
    assert ranked["by_type"][0]["total_ms"] == pytest.approx(2000)


def test_recipe_round_trip_does_not_run(tmp_path):
    recipe = {
        "name": "demo_blend",
        "method": "weighted_sum",
        "sources": [
            {"path": "a.safetensors", "sha256": "abc", "autov2": "aaaaaaaaaa", "weight": 0.6},
            {"path": "b.safetensors", "sha256": "def", "autov2": "bbbbbbbbbb", "weight": 0.4},
        ],
        "notes": "round trip",
    }
    again = merge_recipe.recipe_from_yaml(merge_recipe.recipe_to_yaml(recipe))
    assert again["name"] == recipe["name"]
    assert again["method"] == "weighted_sum"
    assert again["sources"][0]["weight"] == pytest.approx(0.6)
    assert again["sources"][1]["sha256"] == "def"
    assert again["deleted"] == []
    merge_recipe.save_recipe(tmp_path, recipe)
    replay = merge_recipe.replay_recipe(tmp_path, "demo_blend")
    assert replay["executed"] is False
    assert replay["deleted"] == []
    script = Path(replay["script"]).read_text(encoding="utf-8")
    compile(script, replay["script"], "exec")
    assert "DEVICE = 'cpu'" in script
    assert "GPU merge is refused" in script
    assert "subprocess" not in script
    assert list(tmp_path.rglob("*.safetensors")) == []
    with pytest.raises(ValueError):
        merge_recipe.merge_cpu(tmp_path, "demo_blend", device="cuda")


def test_cpu_merge_weighted_sum_keeps_sources(tmp_path):
    amber = SAMPLE_DIR / "loras" / "style_amber.safetensors"
    teal = SAMPLE_DIR / "loras" / "style_teal.safetensors"
    amber_before = amber.read_bytes()
    teal_before = teal.read_bytes()
    recipe = {
        "name": "cpu_check",
        "method": "weighted_sum",
        "sources": [
            {"path": str(amber), "sha256": "x", "autov2": "y", "weight": 0.25},
            {"path": str(teal), "sha256": "x", "autov2": "y", "weight": 0.75},
        ],
    }
    merge_recipe.save_recipe(tmp_path, recipe)
    result = merge_recipe.merge_cpu(tmp_path, "cpu_check", device="cpu")
    assert result["device"] == "cpu"
    assert result["deleted"] == []
    assert amber.read_bytes() == amber_before
    assert teal.read_bytes() == teal_before
    merged = load_file(result["path"])
    expected = 0.25 * np.asarray([1, 0, 0, 0, 0.5, 0.25, 0.1, 0.05], dtype=np.float32)
    expected += 0.75 * np.asarray([0, 1, 0, 0, 0.2, 0.4, 0.6, 0.8], dtype=np.float32)
    assert np.allclose(merged["lora_unet.weight"], expected)
    loaded = merge_recipe.load_recipe(tmp_path, "cpu_check")
    assert loaded["output"]["sha256"] == result["sha256"]


def test_shelf_life_suggests_without_deleting():
    folder = SAMPLE_DIR / "loras"
    before = sorted(path.name for path in folder.glob("*.safetensors"))
    rows = [hash_index.inspect_file(path) for path in folder.glob("*.safetensors")]
    history = json.loads((SAMPLE_DIR / "history" / "history.json").read_text(encoding="utf-8"))
    usage = provenance.usage_by_lora(provenance.parse_history(history))
    report = shelf_life.analyze(rows, usage, unused_days=30, today="2026-10-31")
    assert report["deleted"] == []
    assert shelf_life.files_still_present(rows)
    assert sorted(path.name for path in folder.glob("*.safetensors")) == before
    by_name = {item["filename"]: item for item in report["suggestions"]}
    assert "unused" not in by_name["style_amber.safetensors"]["reasons"]
    assert "duplicate" in by_name["style_amber.safetensors"]["reasons"]
    assert "unused" in by_name["style_teal.safetensors"]["reasons"]
    assert "unused" in by_name["old_unused.safetensors"]["reasons"]
    assert "missing_metadata" in by_name["mystery_lora.safetensors"]["reasons"]
    assert "duplicate" in by_name["style_amber_retag.safetensors"]["reasons"]
    assert all(item["action"] == "suggest_archive" for item in report["suggestions"])
    sha_groups = [group for group in report["duplicates"] if group["hash_type"] == "sha256"]
    assert sha_groups
    assert all("os.remove" not in item["detail"] for item in report["suggestions"])
