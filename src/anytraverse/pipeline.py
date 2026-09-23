"""The AnyTraverse pipeline."""

import torch

from anytraverse import typing as anyt
from anytraverse.history import EncodingHistory
from anytraverse.interfaces import (
    ImageEncoder,
    PromptAttentionMapping,
    TraversabilityPooler,
    UncertaintyPooler,
)
from anytraverse.preferences import (
    get_prompts,
    parse_trav_pref_syntax,
    update_traversability_preferences,
    validate_traversability_preferences,
)
from anytraverse.roi import RegionOfInterest
from anytraverse.state import AnyTraverseState, Threshold, TraversalState


class AnyTraverse:
    """
    Offroad traversability with a VLM and a human operator in the loop.

    Each call to :meth:`step` produces a traversability map, an uncertainty map
    and a :class:`TraversalState` saying whether the operator should be asked
    for help. Operator feedback is given through :meth:`human_call`.

    Args:
        prompt_attention_mapping (PromptAttentionMapping): Produces one ``(H, W)``
            attention map per prompt.
        image_encoder (ImageEncoder): Encodes the image into a vector used to
            compare scenes.
        traversability_pooler (type[TraversabilityPooler]): Pools the attention
            maps into a traversability map.
        uncertainty_pooler (type[UncertaintyPooler]): Pools the attention maps
            into an uncertainty map.
        init_traversability_preferences (TraversabilityPreferences): Initial
            prompts and weights, e.g. ``{"road": 1.0, "bush": -0.8}``.
        roi (RegionOfInterest): The region in front of the robot to monitor.
        threshold (Threshold): Thresholds for the operator-call decisions.
        similarity_func (SimilarityFunction[Encoding, torch.Tensor]): Similarity
            between two encodings; defaults to cosine similarity.
        history_size (int | None): Maximum number of scenes to remember, or
            ``None`` for unlimited.
    """

    def __init__(
        self,
        prompt_attention_mapping: PromptAttentionMapping,
        image_encoder: ImageEncoder,
        traversability_pooler: type[TraversabilityPooler],
        uncertainty_pooler: type[UncertaintyPooler],
        init_traversability_preferences: anyt.TraversabilityPreferences,
        roi: RegionOfInterest,
        threshold: Threshold,
        similarity_func: anyt.SimilarityFunction[anyt.Encoding, torch.Tensor] = (
            torch.cosine_similarity
        ),
        history_size: int | None = None,
    ) -> None:
        self._prompt_attention_mapping = prompt_attention_mapping
        self._image_encoder = image_encoder
        self._traversability_pooler = traversability_pooler
        self._uncertainty_pooler = uncertainty_pooler
        self._traversability_preferences = dict(
            validate_traversability_preferences(init_traversability_preferences)
        )
        self._roi = roi
        self._threshold = threshold
        self._similarity_func = similarity_func
        self._history = EncodingHistory(similarity_func=similarity_func, max_size=history_size)
        self._traversal_state = TraversalState.OK
        self._ref_scene_encoding: anyt.Encoding | None = None
        self._current_scene_encoding: anyt.Encoding | None = None

    @property
    def traversability_preferences(self) -> anyt.TraversabilityPreferences:
        """The current prompts and weights, as ``{"<prompt>": <weight>, ...}``."""
        return dict(self._traversability_preferences)

    @traversability_preferences.setter
    def traversability_preferences(self, updated: anyt.TraversabilityPreferences) -> None:
        self._traversability_preferences = dict(validate_traversability_preferences(updated))

    @property
    def history(self) -> EncodingHistory:
        """The scene history."""
        return self._history

    @property
    def traversal_state(self) -> TraversalState:
        """The decision taken on the most recent frame."""
        return self._traversal_state

    @property
    def threshold(self) -> Threshold:
        """The thresholds used for operator-call decisions."""
        return self._threshold

    def _create_maps(
        self, image: anyt.Image
    ) -> tuple[list[anyt.PromptAttentionMap], anyt.TraversabilityMap, anyt.UncertaintyMap]:
        """Runs the VLM and both poolers on an image."""
        attention_maps = self._prompt_attention_mapping(
            image, get_prompts(self._traversability_preferences)
        )
        traversability_map = self._traversability_pooler.pool(
            attention_maps, self._traversability_preferences
        )
        uncertainty_map = self._uncertainty_pooler.pool(
            attention_maps, self._traversability_preferences
        )
        return attention_maps, traversability_map, uncertainty_map

    def _register_current_scene(
        self, delta_tau: anyt.TraversabilityPreferences | None = None
    ) -> anyt.HistoryElement[anyt.Encoding]:
        """Makes the current scene the reference and stores it in the history."""
        if self._current_scene_encoding is None:
            raise RuntimeError("Call step() at least once before registering a scene")
        if delta_tau:
            self.traversability_preferences = update_traversability_preferences(
                self._traversability_preferences, delta_tau
            )
        self._ref_scene_encoding = self._current_scene_encoding
        return self._history.add(self._current_scene_encoding, self._traversability_preferences)

    def register_scene(self) -> anyt.HistoryElement[anyt.Encoding]:
        """
        Stores the current scene in the history without changing the preferences.

        Returns:
            HistoryElement[Encoding]: The stored element.
        """
        return self._register_current_scene()

    def human_call(self, human_input: str = "") -> anyt.HistoryElement[anyt.Encoding]:
        """
        Applies operator feedback and registers the current scene.

        Args:
            human_input (str): Preference updates in the syntax
                ``"<prompt1>: <weight1>; <prompt2>: <weight2>"``. An empty
                string registers the scene without changing preferences.

        Returns:
            HistoryElement[Encoding]: The stored element.
        """
        delta_tau = parse_trav_pref_syntax(human_input)
        return self._register_current_scene(delta_tau)

    @torch.inference_mode()
    def step(self, image: anyt.Image) -> AnyTraverseState:
        """
        Processes one frame.

        Args:
            image (Image): The RGB camera frame.

        Returns:
            AnyTraverseState: Maps, region-of-interest statistics and the
                :class:`TraversalState` decision for this frame.
        """
        self._current_scene_encoding = self._image_encoder(image)
        if self._ref_scene_encoding is None:
            self._register_current_scene()
        assert self._ref_scene_encoding is not None

        attention_maps, traversability_map, uncertainty_map = self._create_maps(image)
        traversability_roi, roi_start, roi_end = self._roi.extract(traversability_map)
        uncertainty_roi, _, _ = self._roi.extract(uncertainty_map)
        roi_uncertainty = float(uncertainty_roi.mean())

        ref_scene_similarity = float(
            self._similarity_func(
                self._current_scene_encoding,
                self._ref_scene_encoding.to(self._current_scene_encoding),
            ).ravel()[0]
        )

        if ref_scene_similarity < self._threshold.ref_scene_similarity:
            (best_encoding, best_preferences), best_similarity = self._history.find_best_match(
                self._current_scene_encoding
            )
            if best_similarity >= self._threshold.ref_scene_similarity:
                self._ref_scene_encoding = best_encoding
                self.traversability_preferences = update_traversability_preferences(
                    self._traversability_preferences, best_preferences
                )
                self._traversal_state = TraversalState.OK
            else:
                self._traversal_state = TraversalState.UNKNOWN_SCENE
        elif roi_uncertainty > self._threshold.roi_uncertainty:
            self._traversal_state = TraversalState.UNKNOWN_OBJECT
        else:
            self._traversal_state = TraversalState.OK

        return AnyTraverseState(
            image_encoding=self._current_scene_encoding,
            attention_maps=attention_maps,
            traversability_map=traversability_map,
            uncertainty_map=uncertainty_map,
            traversability_map_roi=traversability_roi,
            uncertainty_map_roi=uncertainty_roi,
            ref_scene_similarity=ref_scene_similarity,
            roi_bbox=(roi_start, roi_end),
            roi_traversability=float(traversability_roi.mean()),
            roi_uncertainty=roi_uncertainty,
            traversal_state=self._traversal_state,
            traversability_preferences=self.traversability_preferences,
        )


__all__ = ["AnyTraverse"]
