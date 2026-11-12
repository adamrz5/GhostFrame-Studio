"""
Two-stage identity tracking across video frames.

Stage 1 — Spatial / IoU (in-scene):
    If a face in frame N overlaps enough (IoU) with a face from a recent frame,
    it's the same person — no embedding comparison needed. Fast and robust for
    continuous motion.

Stage 2 — Embedding similarity (cross-scene re-id):
    If IoU gives no match (new scene, cut, re-entry), compare the ArcFace
    embedding against all known persons' mean embeddings using FAISS (or scipy
    as fallback). This is how we recognize someone who left and came back.

Sources that informed this design:
- face-reidentification (yakhyo): FAISS index.search() for batch embedding lookup
- DeepSORT: IoU + appearance for combined matching
- insightface issues: rolling mean embedding improves stability
"""
from __future__ import annotations

import numpy as np

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

from scipy.spatial.distance import cosine as cosine_distance


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _iou(box_a: list[int], box_b: list[int]) -> float:
    """Intersection over Union for [x1,y1,x2,y2] boxes."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    try:
        return float(1.0 - cosine_distance(a, b))
    except Exception:
        return 0.0


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-10 else v


# ─── Person model ─────────────────────────────────────────────────────────────

class Person:
    MAX_EMBEDDINGS = 60  # rolling window

    def __init__(self, person_id: int, thumbnail: np.ndarray):
        self.person_id = person_id
        self.embeddings: list[np.ndarray] = []
        self.thumbnail: np.ndarray = thumbnail      # BGR face crop
        self.frame_data: dict[int, dict] = {}       # {frame_idx: {bbox, det_score, interpolated?}}
        self._is_manual_split: bool = False          # user separated this identity explicitly

    @property
    def mean_embedding(self) -> np.ndarray:
        if not self.embeddings:
            return np.zeros(512, dtype=np.float32)
        return _l2_normalize(np.mean(self.embeddings, axis=0))

    def add_observation(
        self,
        frame_idx: int,
        bbox: list[int],
        det_score: float,
        embedding: np.ndarray,
    ):
        self.frame_data[frame_idx] = {"bbox": bbox, "det_score": det_score}
        self.embeddings.append(embedding)
        if len(self.embeddings) > self.MAX_EMBEDDINGS:
            self.embeddings = self.embeddings[-self.MAX_EMBEDDINGS:]

    @property
    def frame_count(self) -> int:
        return len(self.frame_data)

    def last_bbox_before(self, frame_idx: int, window: int = 30) -> list[int] | None:
        """Return the most recent bbox within `window` frames before frame_idx."""
        for fi in range(frame_idx - 1, max(-1, frame_idx - window - 1), -1):
            if fi in self.frame_data:
                return self.frame_data[fi]["bbox"]
        return None


# ─── Tracker ──────────────────────────────────────────────────────────────────

class FaceTracker:
    def __init__(
        self,
        similarity_threshold: float = 0.65,
        iou_threshold: float = 0.40,
        max_interp_gap: int = 30,
    ):
        self.similarity_threshold = similarity_threshold
        self.iou_threshold = iou_threshold
        self.max_interp_gap = max_interp_gap
        self.persons: list[Person] = []
        self._faiss_index = None           # rebuilt on demand
        self._faiss_index_map: list[int] = []  # FAISS row → self.persons index
        self._next_id: int = 0             # monotonic counter — O(1) vs O(n) max()
        self.scene_cuts: set[int] = set()  # frame indices where a scene cut was detected

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _invalidate_index(self):
        self._faiss_index = None
        self._faiss_index_map = []

    def _next_person_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def _build_faiss_index(self):
        if not _FAISS_AVAILABLE or not self.persons:
            return
        # Solo incluir personas con embeddings reales — un vector cero hace que
        # faiss.normalize_L2() produzca NaN/inf (divisón entre ~0).
        valid = [(i, p) for i, p in enumerate(self.persons) if p.embeddings]
        if not valid:
            return
        dim = len(valid[0][1].mean_embedding)
        index = faiss.IndexFlatIP(dim)  # inner product = cosine sim after L2 norm
        matrix = np.stack([p.mean_embedding for _, p in valid]).astype(np.float32)
        faiss.normalize_L2(matrix)
        index.add(matrix)
        self._faiss_index = index
        self._faiss_index_map = [i for i, _ in valid]  # FAISS fila → índice en self.persons

    def _embedding_match(
        self,
        embedding: np.ndarray,
        excluded_person_ids: set[int] | None = None,
    ) -> tuple[int, float]:
        """
        Find best matching person via FAISS (or scipy fallback).
        Returns (person_list_idx, similarity). person_list_idx=-1 if no match.
        """
        if not self.persons:
            return -1, 0.0
        excluded_person_ids = excluded_person_ids or set()

        norm_emb = _l2_normalize(embedding).astype(np.float32)

        # ── FAISS path ───────────────────────────────────────────────────────
        if _FAISS_AVAILABLE:
            if self._faiss_index is None:
                self._build_faiss_index()
            if self._faiss_index is not None:
                q = norm_emb.reshape(1, -1)
                faiss.normalize_L2(q)
                k = len(self._faiss_index_map) if excluded_person_ids else 1
                sims, idxs = self._faiss_index.search(q, k)
                for sim_raw, idx_raw in zip(sims[0], idxs[0]):
                    faiss_row = int(idx_raw)
                    if faiss_row < 0 or faiss_row >= len(self._faiss_index_map):
                        continue
                    list_idx = self._faiss_index_map[faiss_row]  # FAISS fila → persons
                    if self.persons[list_idx].person_id in excluded_person_ids:
                        continue
                    return list_idx, float(sim_raw)
                return -1, 0.0

        # ── scipy fallback ────────────────────────────────────────────────────
        best_idx, best_sim = -1, -1.0
        for i, person in enumerate(self.persons):
            if person.person_id in excluded_person_ids:
                continue
            sim = _cosine_sim(norm_emb, person.mean_embedding)
            if sim > best_sim:
                best_sim, best_idx = sim, i
        return best_idx, best_sim

    def _iou_match(
        self,
        bbox: list[int],
        frame_idx: int,
        excluded_person_ids: set[int] | None = None,
    ) -> int | None:
        """
        Return person list-index if a person has a recent bbox with IoU >= threshold.
        Checks up to 8 preceding frames (handles skipped frames during analysis).
        """
        best_idx = None
        best_iou = 0.0
        excluded_person_ids = excluded_person_ids or set()
        window = max(8, self.max_interp_gap)
        for i, person in enumerate(self.persons):
            if person.person_id in excluded_person_ids:
                continue
            prev_bbox = person.last_bbox_before(frame_idx, window=window)
            if prev_bbox is None:
                continue
            iou = _iou(bbox, prev_bbox)
            if iou >= self.iou_threshold and iou > best_iou:
                best_iou = iou
                best_idx = i
        return best_idx

    @staticmethod
    def _crop_face(frame_bgr: np.ndarray, bbox: list[int]) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            return frame_bgr[y1:y2, x1:x2].copy()
        return np.zeros((64, 64, 3), dtype=np.uint8)

    # ── Public API ────────────────────────────────────────────────────────────

    def process_frame(
        self,
        frame_idx: int,
        detections: list[dict],
        frame_bgr: np.ndarray,
    ) -> list[int]:
        """
        Update tracker with all detections from one frame.
        Returns list of person_ids in the same order as detections.
        person_id == -1 means embedding was unavailable (detection skipped).
        """
        assigned = []
        used_person_ids: set[int] = set()

        for det in detections:
            emb = det.get("embedding")
            bbox = det.get("bbox", [0, 0, 0, 0])
            score = det.get("det_score", 1.0)

            if emb is None:
                assigned.append(-1)
                continue

            # Stage 1: IoU match (fast, in-scene)
            list_idx = self._iou_match(bbox, frame_idx, used_person_ids)

            # Stage 2: embedding similarity (cross-scene re-id)
            if list_idx is None:
                list_idx, sim = self._embedding_match(emb, used_person_ids)
                if sim < self.similarity_threshold:
                    list_idx = None  # genuinely new person

            if list_idx is not None:
                person = self.persons[list_idx]
            else:
                # Register new person — embedding added below via add_observation
                crop = self._crop_face(frame_bgr, bbox)
                person = Person(self._next_person_id(), crop)
                self.persons.append(person)

            person.add_observation(frame_idx, bbox, score, emb)
            used_person_ids.add(person.person_id)
            assigned.append(person.person_id)

        # Invalidate only when this frame assigned at least one real person.
        # Empty frames should not force a FAISS rebuild on the next detection.
        if assigned and any(pid != -1 for pid in assigned):
            self._invalidate_index()
        return assigned

    def interpolate_bboxes(self):
        """
        Linearly interpolate bounding boxes for frames inside gaps where a person
        wasn't detected. Gaps larger than max_interp_gap are left empty.

        Gaps that cross a detected scene cut (stored in self.scene_cuts) are
        skipped — interpolating across a cut would produce "floating" bboxes
        that don't correspond to any real frame content.
        """
        for person in self.persons:
            keys = sorted(person.frame_data.keys())
            for i in range(len(keys) - 1):
                f0, f1 = keys[i], keys[i + 1]
                gap = f1 - f0
                if gap <= 1 or gap > self.max_interp_gap:
                    continue
                # Skip interpolation if a scene cut falls in this gap
                if self.scene_cuts and any(f0 < cut <= f1 for cut in self.scene_cuts):
                    continue
                b0 = person.frame_data[f0]["bbox"]
                b1 = person.frame_data[f1]["bbox"]
                for j in range(1, gap):
                    t = j / gap
                    interp = [int(b0[k] + t * (b1[k] - b0[k])) for k in range(4)]
                    person.frame_data[f0 + j] = {
                        "bbox": interp,
                        "det_score": 0.5,
                        "interpolated": True,
                    }

    def merge_persons(self, pid_keep: int, pid_discard: int, force: bool = False):
        """
        Merge pid_discard into pid_keep. Rebuilds FAISS index.

        Raises ValueError si las dos personas tienen detecciones reales en el mismo frame
        (no pueden ser la misma persona física), a menos que force=True.
        La UI ya avisa al usuario antes de llamar; este check protege llamadas directas.
        """
        if pid_keep == pid_discard:
            return
        pa = next((p for p in self.persons if p.person_id == pid_keep), None)
        pb = next((p for p in self.persons if p.person_id == pid_discard), None)
        if pa is None or pb is None:
            return
        if not force and self._has_temporal_conflict(pa, pb):
            raise ValueError(
                f"Persona {pid_keep} y persona {pid_discard} aparecen en el mismo frame "
                "como detecciones reales — no pueden ser la misma persona. "
                "Usa force=True para forzar la fusión."
            )
        pa.embeddings.extend(pb.embeddings)
        if len(pa.embeddings) > Person.MAX_EMBEDDINGS:
            pa.embeddings = pa.embeddings[-Person.MAX_EMBEDDINGS:]
        # Prefer real detections over interpolated ones: only overwrite pa's frame
        # if pa has an interpolated entry there (or no entry at all).
        for fi, data in pb.frame_data.items():
            if fi not in pa.frame_data or pa.frame_data[fi].get("interpolated", False):
                pa.frame_data[fi] = data
        self.persons = [p for p in self.persons if p.person_id != pid_discard]
        self._invalidate_index()

    def split_person(self, pid: int, frame_indices: list[int]) -> int:
        """
        Move frame_indices out of person pid into a new person.
        Returns the new person_id, or -1 on failure.

        Manual splits are kept out of automatic consolidation. The user has
        explicitly said "this segment is a different identity", so re-grouping
        must not merge it back only because nearby embeddings are similar.
        """
        original = next((p for p in self.persons if p.person_id == pid), None)
        if original is None or not frame_indices:
            return -1
        new_person = Person(self._next_person_id(), original.thumbnail.copy())
        # No copiar embeddings del original: el segmento dividido tiene su propia identidad.
        # Si se re-consolida tras un split manual, el nuevo no se fusionará de vuelta
        # con el original por similitud de embeddings (serían idénticos si se copiaran).
        new_person.embeddings = []
        new_person._is_manual_split = True
        for fi in frame_indices:
            if fi in original.frame_data:
                new_person.frame_data[fi] = original.frame_data.pop(fi)
        if not new_person.frame_data:
            return -1
        self.persons.append(new_person)
        self._invalidate_index()
        return new_person.person_id

    @staticmethod
    def _has_temporal_conflict(pa: Person, pb: Person) -> bool:
        """
        Two identities seen as real detections in the same frame cannot be the
        same physical person, even if their embeddings are close.
        """
        a_frames = {
            fi for fi, data in pa.frame_data.items()
            if not data.get("interpolated", False)
        }
        b_frames = {
            fi for fi, data in pb.frame_data.items()
            if not data.get("interpolated", False)
        }
        return bool(a_frames.intersection(b_frames))

    def consolidate_persons(self, sim_threshold: float = 0.60) -> int:
        """
        Post-analysis clustering pass: merge persons whose mean embeddings are
        similar enough. Fixes fragmentation where the same real person got split
        into multiple IDs due to lighting changes, camera angles or scene cuts.

        Uses O(n²) pairwise cosine similarity (fast enough for typical n < 50).
        Merges the person with fewer frames into the one with more frames.
        Call after interpolate_bboxes(), before showing results to the user.

        Returns the number of merges performed.
        """
        merges = 0
        changed = True
        while changed:
            changed = False
            n = len(self.persons)
            for i in range(n):
                for j in range(i + 1, n):
                    pa = self.persons[i]
                    pb = self.persons[j]
                    if getattr(pa, "_is_manual_split", False) or getattr(pb, "_is_manual_split", False):
                        continue
                    if self._has_temporal_conflict(pa, pb):
                        continue
                    sim = _cosine_sim(pa.mean_embedding, pb.mean_embedding)
                    if sim >= sim_threshold:
                        # Keep the person with more observations as the canonical ID
                        if pb.frame_count >= pa.frame_count:
                            self.merge_persons(pb.person_id, pa.person_id)
                        else:
                            self.merge_persons(pa.person_id, pb.person_id)
                        merges += 1
                        changed = True
                        break  # list changed — restart inner loop
                if changed:
                    break
        return merges

    def prune_persons(self, min_real_frames: int = 5) -> int:
        """
        Remove persons detected in fewer than min_real_frames non-interpolated
        frames. These are almost certainly false positives: reflections, blurry
        background faces, or partial detections that briefly crossed the detector
        threshold.

        Call after consolidate_persons().
        Returns the number of persons removed.
        """
        before = len(self.persons)
        self.persons = [
            p for p in self.persons
            if sum(
                1 for v in p.frame_data.values()
                if not v.get("interpolated", False)
            ) >= min_real_frames
        ]
        removed = before - len(self.persons)
        if removed:
            self._invalidate_index()
        return removed
