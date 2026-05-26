from __future__ import annotations

import numpy as np

from core.face_tracker import FaceTracker, Person


def _embedding(value: float = 1.0) -> np.ndarray:
    vec = np.full(512, value, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def _person(pid: int, frame_idx: int = 0, interpolated: bool = False) -> Person:
    person = Person(pid, np.zeros((8, 8, 3), dtype=np.uint8))
    data = {"bbox": [1, 1, 6, 6], "det_score": 0.9}
    if interpolated:
        data["interpolated"] = True
    person.frame_data[frame_idx] = data
    person.embeddings = [_embedding()]
    return person


def test_merge_persons_real_detection_survives_interpolated_conflict():
    tracker = FaceTracker()
    keep = _person(0, frame_idx=10, interpolated=True)
    discard = _person(1, frame_idx=10, interpolated=False)
    discard.frame_data[10]["bbox"] = [2, 2, 7, 7]
    tracker.persons = [keep, discard]

    tracker.merge_persons(0, 1)

    assert tracker.persons[0].frame_data[10]["bbox"] == [2, 2, 7, 7]
    assert not tracker.persons[0].frame_data[10].get("interpolated", False)


def test_merge_persons_embedding_list_is_capped():
    tracker = FaceTracker()
    keep = _person(0)
    discard = _person(1, frame_idx=20)
    keep.embeddings = [_embedding(1.0) for _ in range(Person.MAX_EMBEDDINGS)]
    discard.embeddings = [_embedding(2.0) for _ in range(10)]
    tracker.persons = [keep, discard]

    tracker.merge_persons(0, 1)

    assert len(tracker.persons[0].embeddings) == Person.MAX_EMBEDDINGS


def test_split_person_moved_frames_no_longer_in_original():
    tracker = FaceTracker()
    person = _person(0, frame_idx=1)
    person.frame_data[2] = {"bbox": [2, 2, 7, 7], "det_score": 0.8}
    tracker.persons = [person]

    new_pid = tracker.split_person(0, [2])

    assert new_pid != -1
    assert 2 not in person.frame_data
    assert tracker.persons[-1].frame_data[2]["bbox"] == [2, 2, 7, 7]


def test_consolidate_persons_identical_embeddings_are_merged():
    tracker = FaceTracker()
    first = _person(0, frame_idx=1)
    second = _person(1, frame_idx=20)
    second.embeddings = [first.embeddings[0].copy()]
    tracker.persons = [first, second]

    merges = tracker.consolidate_persons(sim_threshold=0.99)

    assert merges == 1
    assert len(tracker.persons) == 1


def test_has_temporal_conflict_true_for_same_real_frame():
    first = _person(0, frame_idx=3)
    second = _person(1, frame_idx=3)

    assert FaceTracker._has_temporal_conflict(first, second) is True
