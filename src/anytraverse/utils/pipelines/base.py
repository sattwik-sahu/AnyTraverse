from typing import Type
import torch
from anytraverse import typing as anyt
from anytraverse.utils.base.attention_mapping import PromptAttentionMapping
from anytraverse.utils.base.map_pooling import TraversabilityPooler, UncertaintyPooler
from anytraverse.utils.roi import RegionOfInterest
from anytraverse.utils.state import TraversalState, Threshold, AnyTraverseState
from anytraverse.utils.base.history import EncodingHistory
from anytraverse.utils.base.encoding import ImageEncoder
from anytraverse.utils.trav_pref import (
    update_traversability_preferences,
    parse_trav_pref_syntax,
    get_prompts,
    get_weights,
)


class AnyTraverse[TImage: anyt.Image]:
    """
    The offroad traversability pipeline discussed in "AnyTraverse: VLM-Based Offroad Traversability Framework with Human-in-the-Loop".
    """

    def __init__(
        self,
        prompt_attention_mapping: PromptAttentionMapping[TImage],
        traversability_pooler: Type[TraversabilityPooler],
        uncertainty_pooler: Type[UncertaintyPooler],
        init_traversability_preferences: anyt.TraversabilityPreferences,
        image_encoder: ImageEncoder[TImage],
        similarity_func: anyt.SimilarityFunction[anyt.Encoding, torch.Tensor],
        roi: RegionOfInterest,
        threshold: Threshold,
    ) -> None:
        """
        Initlializes the AnyTraverse pipeline for traversability segmentation.

        Args:
            prompt_attention_mapping (PromptAttentionMapping[TImage]):
                A `PromptAttentionMapping` module that takes in an image of type `TImage`
                and the prompts and outputs the `AttentionMap` for each prompt on the image.
            traversability_pooler (Type[TraversabilityPooler]):
                A pooler that takes in the attention maps and pools them into a traversability
                map using the provided traversability preferences.
            uncertainty_pooler (Type[UncertaintyPooler]):
                A pooler that takes in the attention maps and pools them into an uncertainty
                map using the provided traversability preferences.
            init_traversability_preferences (anyt.TraversabilityPreferences):
                The initial traversability preferences expressed as a `dict[str, float]`
                of the form `{"<prompt1>": <weight1>, ...}`
            image_encoder (ImageEncoder[TImage]):
                An image encoder that takes in and image of type `TImage` and outputs the
                encoded image as an `anyt.Encoding` (alias for `torch.Tensor`)
            similarity_func (SimilarityFunc[anyt.Encoding, torch.Tensor]):
                The simiilarity function to be used to match image encodings in the history.
                It should accept two `anyt.Encoding` encodings and output the similarity as
                a `torch.Tensor`.
                The type is an alias for `Callable[[anyt.Encoding, anyt.Encoding], torch.Tensor]`
            roi (RegionOfInterest):
                The `RegionOfInterest` module to extract the ROI in required maps.
            threshold (Threshold):
                The `dataclass` `Threshold`, containing the thresholds discussed in the paper
                - The reference scene similarity threshold (`ref_scene_sim`) for unseen scene human operator calls.
                - The ROI uncertainty threshold (`roi_uncertainty`) for unknown object human operator calls.
        """
        self._prompt_attention_mapping = prompt_attention_mapping
        self._traversability_pooler = traversability_pooler
        self._uncertainty_pooler = uncertainty_pooler
        self._traversability_preferences = init_traversability_preferences
        self._roi = roi
        self._similarity_func = similarity_func
        self._history = EncodingHistory(similarity_func=similarity_func)
        self._image_encoder = image_encoder
        self._traversal_state = TraversalState.OK
        self._threshold = threshold
        self._ref_scene_encoding: anyt.Encoding = torch.empty(0)
        self._current_scene_encoding: anyt.Encoding = torch.empty(0)

    @property
    def traversability_preferences(self) -> anyt.TraversabilityPreferences:
        """
        The traversability preferences as a `dict[str, float]` in
        the form `{"<prompt1>": <weight1>, ...}`
        """
        return self._traversability_preferences

    @traversability_preferences.setter
    def traversability_preferences(
        self, updated_traversability_preferences: anyt.TraversabilityPreferences
    ) -> anyt.TraversabilityPreferences:
        prompts, weights = (
            get_prompts(updated_traversability_preferences),
            get_weights(updated_traversability_preferences),
        )
        assert all(len(prompt) > 0 for prompt in prompts), (
            "Empty prompts not allowed in traversability preferences"
        )
        assert all(weight >= -1 and weight <= 1 for weight in weights), (
            "All weights in traversability preferences must be in range [-1, 1]"
        )
        self._traversability_preferences = updated_traversability_preferences
        return self._traversability_preferences

    def _create_maps(
        self, image: TImage
    ) -> tuple[
        list[anyt.PromptAttentionMap], anyt.TraversabilityMap, anyt.UncertaintyMap
    ]:
        """
        Creates the different maps to be used in the pipeline.

        Args:
            image (TImage): The image to generate the maps from.

        Returns:
            - attention_maps (list[PromptAttentionMap]):
                The attention maps on the image for each of the prompts, as a list.
            - traversability_map (TraversabilityMap):
                The traversability map showing the traversability (0-1) of each pixel in the image.
            - uncertainty_map (UncertaintyMap):
                The uncertainty map showing the uncertanty (0-1) of each pixel in the image.

            Each of the maps are `torch.Tensor` with shape `(H, W)`, the same as height and width of the image.
        """
        attention_maps = self._prompt_attention_mapping(
            x=image,
            prompts=get_prompts(
                traversability_preferences=self._traversability_preferences
            ),
        )
        traversability_map = self._traversability_pooler.pool(
            maps=attention_maps,
            traversability_preferences=self._traversability_preferences,
        )
        uncertainty_map = self._uncertainty_pooler.pool(
            maps=attention_maps,
            traversability_preferences=self._traversability_preferences,
        )
        return attention_maps, traversability_map, uncertainty_map

    def _update_traversability_preferences(
        self, delta_tau: anyt.TraversabilityPreferences
    ) -> anyt.TraversabilityPreferences:
        self._traversability_preferences = update_traversability_preferences(
            traversability_preferences=self._traversability_preferences,
            updates=delta_tau,
        )
        return self._traversability_preferences

    def _update_tau_and_history(
        self, delta_tau: anyt.TraversabilityPreferences = dict()
    ) -> anyt.HistoryElement[anyt.Encoding]:
        # Set reference scene encoding to current scene encoding
        self._ref_scene_encoding = self._current_scene_encoding
        # NOTE Technical - load to right device
        self._ref_scene_encoding = self._ref_scene_encoding.to(
            device=self._current_scene_encoding.device
        )
        # Update traversability preferences
        self._traversability_preferences = update_traversability_preferences(
            traversability_preferences=self._traversability_preferences,
            updates=delta_tau,
        )
        # Save new element in history
        new_history_element = self._history.add(
            key=self._current_scene_encoding,
            traversabilty_preferences=self._traversability_preferences,
        )
        return new_history_element

    def human_call(self, human_input: str) -> anyt.HistoryElement[anyt.Encoding]:
        """
        Perform a human operator call with human inputs for the traversability
        preferences expressed in the syntax:
        `<prompt1>: <weight1>[; <prompt2>: <weight2>; ...]`

        Args:
            human_input (str): The human input as a `str` in the syntax above.

        Returns:
            The updated traversability preferences in the AnyTraverse pipeline.
        """
        delta_tau = parse_trav_pref_syntax(syntax=human_input)
        return self._update_tau_and_history(delta_tau=delta_tau)

    def step(self, image: TImage) -> AnyTraverseState:
        """
        Perform one time step in the AnyTraverse pipeline, on the image of type `TImage`.
        The actual logic explained in the paper is implemented here.

        Args:
            image (TImage): The image to perform the functionality on.

        Returns:
            AnyTraverseState:
                The state of after passing the `image` through the AnyTraverse pipeline.
        """
        # Get the encoding of the current scene
        self._current_scene_encoding = self._image_encoder(x=image)

        # Set the reference scene encoding to the encoding of the first frame
        if self._ref_scene_encoding.numel() == 0:
            self._update_tau_and_history()

        # Create the different maps
        attention_maps, traversability_map, uncertainty_map = self._create_maps(
            image=image
        )

        # Extract the ROI from the traversability and uncertainty maps
        traversability_map_roi, roi_bbox_start, roi_bbox_end = self._roi.extract(
            mat=traversability_map
        )
        uncertainty_map_roi, _, _ = self._roi.extract(mat=uncertainty_map)

        # Check if human operator call required
        # Get the similarity with the reference scene
        ref_scene_similarity = self._similarity_func(
            self._current_scene_encoding, self._ref_scene_encoding
        )
        # Current scene not sufficiently similar to reference scene
        if ref_scene_similarity < self._threshold.ref_scene_similarity:
            # Search history for best match
            best_match_encoding, best_match_traversability_preferences = (
                self._history.find_best_match(query=self._current_scene_encoding)
            )
            # Is the best match similar enough?
            best_match_similarity = self._similarity_func(
                self._current_scene_encoding, best_match_encoding
            )
            if best_match_similarity >= self._threshold.ref_scene_similarity:
                # Successful hit in history
                # Update the reference scene embedding
                self._ref_scene_encoding = best_match_encoding
                # Update traversability preferences according to match in history
                self._update_traversability_preferences(
                    delta_tau=best_match_traversability_preferences
                )
                # Avoided redundant human operator call using history, good to go!
                self._traversal_state = TraversalState.OK
            else:
                # No good match found in history, unknown scene encountered
                self._traversal_state = TraversalState.UNKNOWN_SCENE
        elif uncertainty_map_roi.mean() > self._threshold.roi_uncertainty:
            # Too much uncertainty in the image ROI, unknown object detected
            self._traversal_state = TraversalState.UNKOWN_OBJ
        else:
            # No human operator call required, good to go!
            self._traversal_state = TraversalState.OK

        return AnyTraverseState(
            image_encoding=self._current_scene_encoding,
            attention_maps=attention_maps,
            traversability_map=traversability_map,
            uncertainty_map=uncertainty_map,
            traversability_map_roi=traversability_map_roi,
            uncertainty_map_roi=uncertainty_map_roi,
            ref_scene_similarity=ref_scene_similarity.item(),
            roi_bbox=(roi_bbox_start, roi_bbox_end),
            roi_traversability=traversability_map_roi.mean().item(),
            roi_uncertainty=uncertainty_map_roi.mean().item(),
            traversal_state=self._traversal_state,
            traversability_preferences=self.traversability_preferences,
        )
