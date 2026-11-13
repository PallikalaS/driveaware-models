import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from driveaware_data.crop import Box, dynamic_crop, upper_body_box  # noqa: E402
from driveaware_data.labels import (APP_CLASSES, SUBCLASS_TO_APP, SUBCLASSES, assign_split,  # noqa: E402
                                    map_public_label, parse_clip_name)


def test_every_subclass_maps_to_an_app_class():
    assert set(SUBCLASS_TO_APP) == set(SUBCLASSES)
    assert set(SUBCLASS_TO_APP.values()) == set(APP_CLASSES)


def test_clip_names():
    c = parse_clip_name("bindu_texting_night-clear_t1.mp4")
    assert (c.name, c.activity, c.condition, c.take) == ("bindu", "texting", "night-clear", 1)
    assert parse_clip_name("swathi_phone_in_hand_day-sun_t2.mp4").activity == "phone_in_hand"
    with pytest.raises(ValueError):
        parse_clip_name("Bindu_Texting_night_t1.mp4")


def test_held_out_is_test_only():
    held = {"mahitha", "vanessa"}
    assert assign_split("Mahitha", held, set()) == "test"
    assert assign_split("bindu", held, {"bindu"}) == "val"
    assert assign_split("swathi", held, {"bindu"}) == "train"


def test_public_mapping_drops_unused_classes():
    assert map_public_label("statefarm", "c5") is None
    assert map_public_label("statefarm", "c1") == "texting"


def test_dynamic_crop_keeps_lap_and_is_square():
    face = Box(280, 60, 80, 100)
    b = upper_body_box(face, 640, 480)
    assert b.y + b.h == 480 and b.w == 320  # reaches the lap, 4x face width
    img = np.zeros((480, 640, 3), np.uint8)
    out, used = dynamic_crop(img, face)
    assert used and out.shape == (224, 224, 3)


def test_iphone_mov_clips_are_accepted(tmp_path):
    from driveaware_data.labels import find_videos, parse_clip_name
    assert parse_clip_name("swathi_texting_day-none_t1.MOV").activity == "texting"
    assert parse_clip_name("swathi_safe_night-sun_t1.mov").condition == "night-sun"
    for n in ("a.mp4", "b.MOV", "c.mov", "d.txt", ".hidden.mov"):
        (tmp_path / n).write_bytes(b"")
    assert [p.name for p in find_videos(tmp_path)] == ["a.mp4", "b.MOV", "c.mov"]
