from __future__ import annotations

import base64
import json
import logging
import re
from collections.abc import Iterable

import rq
import requests
from django.conf import settings
from django.db import transaction
from drf_spectacular.utils import OpenApiResponse, extend_schema
from rest_framework import serializers, viewsets
from rest_framework.decorators import action
from rest_framework.exceptions import PermissionDenied, ValidationError
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from cvat.apps.dataset_manager.task import patch_task_data
from cvat.apps.engine.frame_provider import TaskFrameProvider
from cvat.apps.engine.models import (
    FrameQuality,
    LabeledImage,
    LabeledShape,
    RequestAction,
    RequestTarget,
    SourceType,
    Task,
    TrackedShape,
)
from cvat.apps.engine.permissions import TaskPermission, get_iam_context
from cvat.apps.engine.rq import AutoAnnotateRequestId, BaseRQMeta
from cvat.apps.engine.serializers import LabeledDataSerializer
from cvat.apps.engine.types import ExtendedRequest
from cvat.apps.redis_handler.background import AbstractRequestManager
from cvat.apps.redis_handler.serializers import RqIdSerializer

_LOG = logging.getLogger(__name__)
_GEMINI_COORD_SCALE = 1000.0


class ModelAnnotateSerializer(serializers.Serializer):
    task_id = serializers.IntegerField(min_value=1)
    frame_ids = serializers.ListField(
        child=serializers.IntegerField(min_value=0),
        allow_empty=False,
    )
    context_frame_ids = serializers.ListField(
        child=serializers.IntegerField(min_value=0),
        allow_empty=True,
        required=False,
        default=list,
    )
    prompt = serializers.CharField(allow_blank=False, trim_whitespace=True, max_length=8000)
    options = serializers.DictField(required=False, default=dict)

    def validate(self, attrs):
        attrs["frame_ids"] = list(dict.fromkeys(attrs["frame_ids"]))
        attrs["context_frame_ids"] = list(dict.fromkeys(attrs.get("context_frame_ids", [])))

        if len(attrs["frame_ids"]) != 1:
            raise serializers.ValidationError("Exactly one target frame must be provided.")

        max_context = int(settings.GEMINI.get("MAX_CONTEXT_FRAMES", 1))
        if len(attrs["context_frame_ids"]) > max_context:
            raise serializers.ValidationError(
                f"Only one context frame is allowed (configured max is {max_context})."
            )

        if attrs["context_frame_ids"] and attrs["context_frame_ids"][0] == attrs["frame_ids"][0]:
            raise serializers.ValidationError("Context frame must be different from target frame.")

        return attrs


class ModelAnnotateManager(AbstractRequestManager):
    QUEUE_NAME = settings.CVAT_QUEUES.AUTO_ANNOTATION.value
    SUPPORTED_TARGETS = {RequestTarget.TASK}

    def __init__(
        self,
        *,
        request: ExtendedRequest,
        db_instance: Task,
        frame_ids: list[int],
        context_frame_ids: list[int],
        prompt: str,
        options: dict,
    ) -> None:
        super().__init__(request=request, db_instance=db_instance)
        self._frame_ids = frame_ids
        self._context_frame_ids = context_frame_ids
        self._prompt = prompt
        self._options = options

    def build_request_id(self):
        log = logging.getLogger(__name__)

        rq_id =  AutoAnnotateRequestId(
            action=RequestAction.GPTANNOTATE,
            target=RequestTarget.TASK,
            target_id=self.db_instance.pk,
            user_id=self.request.user.id
        ).render()

        log.info("Built model rq id: %s", rq_id)

        return rq_id

    def init_callback_with_params(self) -> None:
        self.callback = run_model_annotation_task
        self.callback_args = (
            self.db_instance.pk,
            self._frame_ids,
            self._context_frame_ids,
            self._prompt,
            self._options,
        )


def _set_current_job_state(*, message: str | None = None, progress: float | None = None) -> None:
    job = rq.get_current_job()
    if not job:
        return

    meta = BaseRQMeta.for_job(job)
    if message is not None:
        meta.status = message
    if progress is not None:
        meta.progress = max(0.0, min(1.0, float(progress)))
    meta.save()


def _extract_json(response_text: str) -> dict | list:
    try:
        payload = json.loads(response_text)
        if isinstance(payload, (dict, list)):
            return payload
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", response_text, re.DOTALL)
    if fenced_match:
        return json.loads(fenced_match.group(1))

    first_object = response_text.find("{")
    last_object = response_text.rfind("}")
    if first_object >= 0 and last_object > first_object:
        try:
            return json.loads(response_text[first_object : last_object + 1])
        except Exception:  # noqa: BLE001
            pass

    first_array = response_text.find("[")
    last_array = response_text.rfind("]")
    if first_array >= 0 and last_array > first_array:
        return json.loads(response_text[first_array : last_array + 1])

    raise ValueError("Model response does not contain valid JSON")


def _extract_gemini_text(payload: dict) -> str:
    collected: list[str] = []
    for candidate in payload.get("candidates", []) or []:
        content = candidate.get("content", {}) or {}
        for part in content.get("parts", []) or []:
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                collected.append(text.strip())
    return "\n".join(collected).strip()


def _label_mapping(task: Task) -> dict[str, int]:
    labels = task.project.label_set.all() if task.project_id else task.label_set.all()
    return {label.name.lower(): label.id for label in labels}


def _to_model_part(frame_provider: TaskFrameProvider, frame_id: int) -> tuple[dict, int, int]:
    frame_data = frame_provider.get_frame(frame_id, quality=FrameQuality.ORIGINAL)
    image_bytes = frame_data.data.getvalue()

    from io import BytesIO

    from PIL import Image

    width, height = Image.open(BytesIO(image_bytes)).size
    model_part = {
        "inline_data": {
            "mime_type": frame_data.mime,
            "data": base64.b64encode(image_bytes).decode("utf-8"),
        }
    }
    return model_part, width, height


def _build_content(
    *,
    user_prompt: str,
    label_names: Iterable[str],
    target_frame_id: int,
    target_part: dict,
    context_parts: list[dict],
) -> list[dict]:
    labels_prompt = ", ".join(sorted(label_names))
    instruction = (
        "You are assisting with image annotation.\n"
        f"Allowed classes for detections: [{labels_prompt}].\n"
        "Return STRICT JSON only in this exact format:\n"
        '[{"label":"object_name","box_2d":[ymin,xmin,ymax,xmax]}]\n'
        "Detection constraints:\n"
        "- box_2d values must be normalized to 0-1000 for the target image\n"
        "- xmin < xmax and ymin < ymax\n"
        "- label must be from allowed classes\n"
        "- If no objects match, return []\n"
        f"User prompt: {user_prompt}\n"
        f"Target frame id: {target_frame_id}."
    )

    content: list[dict] = [{"text": instruction}]

    for idx, context_part in enumerate(context_parts, start=1):
        content.append({"text": f"Context image {idx}"})
        content.append(context_part)

    content.append({"text": "Target image"})
    content.append(target_part)

    return content


def _call_provider(*, content: list[dict], model_name: str, timeout: int) -> tuple[str, dict | list | None]:
    provider_settings = getattr(settings, "GEMINI", {})
    api_key = provider_settings.get("API_KEY", "")
    if not api_key:
        raise ValidationError("Gemini API key is not configured.")

    base_url = str(provider_settings.get("URL", "https://generativelanguage.googleapis.com")).rstrip("/")
    endpoint = f"{base_url}/v1beta/models/{model_name}:generateContent"
    request_body = {
        "contents": [{"role": "user", "parts": content}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.1,
        },
    }

    response = requests.post(
        endpoint,
        params={"key": api_key},
        json=request_body,
        timeout=timeout,
    )
    if not response.ok:
        _LOG.error("Gemini request failed | status=%s body=%s", response.status_code, response.text)
        try:
            detail = response.json()
        except Exception:  # noqa: BLE001
            detail = {"error": response.text}
        raise ValidationError(detail)

    response_payload = response.json()
    text = _extract_gemini_text(response_payload)
    _LOG.info("Model response text: %s", text)
    if not text:
        text = json.dumps(response_payload)
    if not text:
        raise ValueError("Model provider returned an empty response")

    try:
        return text, _extract_json(text)
    except Exception:  # noqa: BLE001
        return text, None


def _normalize_box_2d(raw_bbox: list, width: int, height: int) -> list[float] | None:
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        return None

    try:
        ymin, xmin, ymax, xmax = [float(v) for v in raw_bbox]
    except (TypeError, ValueError):
        return None

    ymin = max(0.0, min(ymin, _GEMINI_COORD_SCALE))
    xmin = max(0.0, min(xmin, _GEMINI_COORD_SCALE))
    ymax = max(0.0, min(ymax, _GEMINI_COORD_SCALE))
    xmax = max(0.0, min(xmax, _GEMINI_COORD_SCALE))

    x1 = (xmin / _GEMINI_COORD_SCALE) * float(width)
    y1 = (ymin / _GEMINI_COORD_SCALE) * float(height)
    x2 = (xmax / _GEMINI_COORD_SCALE) * float(width)
    y2 = (ymax / _GEMINI_COORD_SCALE) * float(height)

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def _normalize_bbox(raw_bbox: list, width: int, height: int) -> list[float] | None:
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        return None

    try:
        x1, y1, x2, y2 = [float(v) for v in raw_bbox]
    except (TypeError, ValueError):
        return None

    x1 = max(0.0, min(x1, float(width)))
    y1 = max(0.0, min(y1, float(height)))
    x2 = max(0.0, min(x2, float(width)))
    y2 = max(0.0, min(y2, float(height)))
    if x2 <= x1 or y2 <= y1:
        return None

    _LOG.info("Before normalization: %s", raw_bbox)
    _LOG.info("After normalization: %s", [x1, y1, x2, y2])

    return [x1, y1, x2, y2]


def _has_any_annotations_on_frame(task_id: int, frame_id: int) -> bool:
    return (
        LabeledShape.objects.filter(job__segment__task_id=task_id, frame=frame_id).exists()
        or LabeledImage.objects.filter(job__segment__task_id=task_id, frame=frame_id).exists()
        or TrackedShape.objects.filter(track__job__segment__task_id=task_id, frame=frame_id).exists()
    )


def run_model_annotation_task(
    task_id: int,
    frame_ids: list[int],
    context_frame_ids: list[int],
    prompt: str,
    options: dict,
) -> int:
    _LOG.info("[MODEL] Starting annotation task=%s frames=%s", task_id, frame_ids)
    _set_current_job_state(message="Preparing task", progress=0.0)

    task = Task.objects.select_related("project", "data").get(pk=task_id)
    frame_provider = TaskFrameProvider(task)
    label_map = _label_mapping(task)

    target_frame_id = frame_ids[0]
    frame_provider.validate_frame_number(target_frame_id)

    timeout = max(1, int(options.get("timeout", settings.GEMINI.get("TIMEOUT", 120))))
    model = str(options.get("model", settings.GEMINI.get("MODEL", "gemini-3-flash-preview")))

    context_parts: list[dict] = []
    if context_frame_ids:
        context_frame_id = context_frame_ids[0]
        frame_provider.validate_frame_number(context_frame_id)
        if not _has_any_annotations_on_frame(task_id, context_frame_id):
            raise ValidationError("The selected context frame has no existing annotations.")

        context_part, _, _ = _to_model_part(frame_provider, context_frame_id)
        context_parts.append(context_part)

    _set_current_job_state(message="Calling provider model", progress=0.25)
    target_part, width, height = _to_model_part(frame_provider, target_frame_id)
    content = _build_content(
        user_prompt=prompt,
        label_names=label_map.keys(),
        target_frame_id=target_frame_id,
        target_part=target_part,
        context_parts=context_parts,
    )

    # Exactly one provider call per request
    raw_response_text, parsed_json = _call_provider(content=content, model_name=model, timeout=timeout)
    _LOG.info("Raw model JSON: %s", parsed_json)
    _LOG.info("Image width=%s height=%s", width, height)

    shapes: list[dict] = []
    detections: list = []
    if isinstance(parsed_json, list):
        detections = parsed_json
    elif isinstance(parsed_json, dict):
        raw_detections = parsed_json.get("detections", [])
        if isinstance(raw_detections, list):
            detections = raw_detections

    for idx, detection in enumerate(detections):
        if not isinstance(detection, dict):
            _LOG.warning("[MODEL] Skipping invalid detection at index=%s: not an object", idx)
            continue

        raw_label = str(detection.get("label", detection.get("class", ""))).strip().lower()
        label_id = label_map.get(raw_label)
        if label_id is None:
            _LOG.warning("[MODEL] Skipping detection with unknown label='%s'", raw_label)
            continue

        bbox = _normalize_box_2d(detection.get("box_2d", []), width, height)
        if bbox is None:
            bbox = _normalize_bbox(detection.get("bbox", []), width, height)
        if bbox is None:
            _LOG.warning("[MODEL] Skipping detection with invalid coordinates=%s", detection)
            continue

        shapes.append(
            {
                "type": "rectangle",
                "label_id": label_id,
                "frame": target_frame_id,
                "points": bbox,
                "occluded": False,
                "outside": False,
                "attributes": [],
                "source": str(SourceType.AUTO),
            }
        )

    if not shapes:
        text_message = raw_response_text.strip().replace("\n", " ")
        if len(text_message) > 500:
            text_message = text_message[:497] + "..."
        _set_current_job_state(message=f"Model response: {text_message}", progress=1.0)
        _LOG.info("[MODEL] Completed without detections for task=%s", task_id)
        return 0

    _set_current_job_state(message="Saving annotations", progress=0.9)
    payload = {"version": 0, "tags": [], "shapes": shapes, "tracks": []}
    serializer = LabeledDataSerializer(
        data=payload,
        context={"annotation_action": "create"},
    )
    serializer.is_valid(raise_exception=True)

    with transaction.atomic():
        patch_task_data(task_id, serializer.validated_data, action="create")

    _set_current_job_state(message="Completed", progress=1.0)
    _LOG.info("[MODEL] Saved %s shapes for task=%s", len(shapes), task_id)
    return len(shapes)


@extend_schema(tags=["model"])
class ModelAnnotationViewSet(viewsets.GenericViewSet):
    permission_classes = [IsAuthenticated]
    serializer_class = ModelAnnotateSerializer
    iam_organization_field = None
    iam_permission_class = TaskPermission

    @extend_schema(
        methods=["POST"],
        summary="Generate annotations with AI model",
        request=ModelAnnotateSerializer,
        responses={
            "202": OpenApiResponse(RqIdSerializer),
            "409": OpenApiResponse(description="Request already running"),
        },
    )
    @action(detail=False, methods=["POST"], url_path="annotate")
    def annotate(self, request: ExtendedRequest):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        try:
            task = Task.objects.select_related("project", "organization").get(pk=data["task_id"])
        except Task.DoesNotExist as ex:
            raise ValidationError(str(ex)) from ex

        iam_context = get_iam_context(request, task)
        perm = TaskPermission.create_base_perm(
            request,
            self,
            TaskPermission.Scopes.UPDATE_ANNOTATIONS,
            iam_context,
            task,
        )
        result = perm.check_access()
        if not result.allow:
            raise PermissionDenied(", ".join(result.reasons) if result.reasons else None)

        if data["context_frame_ids"]:
            context_frame_id = data["context_frame_ids"][0]
            if not _has_any_annotations_on_frame(task.id, context_frame_id):
                raise ValidationError("The selected context frame has no existing annotations.")

        manager = ModelAnnotateManager(
            request=request,
            db_instance=task,
            frame_ids=data["frame_ids"],
            context_frame_ids=data["context_frame_ids"],
            prompt=data["prompt"],
            options=data.get("options", {}),
        )
        return manager.enqueue_job()
