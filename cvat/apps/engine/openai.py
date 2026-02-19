from __future__ import annotations

import base64
import json
import logging
import re
from collections.abc import Iterable

import requests
import rq
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


class OpenAIAnnotateSerializer(serializers.Serializer):
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

        max_context = int(settings.OPENAI.get("MAX_CONTEXT_FRAMES", 1))
        if len(attrs["context_frame_ids"]) > max_context:
            raise serializers.ValidationError(
                f"Only one context frame is allowed (configured max is {max_context})."
            )

        if attrs["context_frame_ids"] and attrs["context_frame_ids"][0] == attrs["frame_ids"][0]:
            raise serializers.ValidationError("Context frame must be different from target frame.")

        return attrs


class OpenAIAnnotateManager(AbstractRequestManager):
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
            action=RequestAction.OPENAIANNOTATE,
            target=RequestTarget.TASK,
            target_id=self.db_instance.pk,
            user_id=self.request.user.id
        ).render()

        log.info("Built OpenAI rq id: %s", rq_id)

        return rq_id

    def init_callback_with_params(self) -> None:
        self.callback = run_openai_annotation_task
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


def _extract_json(response_text: str) -> dict:
    try:
        payload = json.loads(response_text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", response_text, re.DOTALL)
    if fenced_match:
        return json.loads(fenced_match.group(1))

    first_brace = response_text.find("{")
    last_brace = response_text.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        return json.loads(response_text[first_brace : last_brace + 1])

    raise ValueError("OpenAI response does not contain valid JSON")


def _extract_output_text(payload: dict) -> str:
    if output_text := payload.get("output_text"):
        return str(output_text).strip()

    collected: list[str] = []
    for output_item in payload.get("output", []) or []:
        for content_item in output_item.get("content", []) or []:
            text = content_item.get("text")
            if isinstance(text, str) and text.strip():
                collected.append(text.strip())

    return "\n".join(collected).strip()


def _label_mapping(task: Task) -> dict[str, int]:
    labels = task.project.label_set.all() if task.project_id else task.label_set.all()
    return {label.name.lower(): label.id for label in labels}


def _to_image_data_url(frame_provider: TaskFrameProvider, frame_id: int) -> tuple[str, int, int]:
    frame_data = frame_provider.get_frame(frame_id, quality=FrameQuality.ORIGINAL)
    image_bytes = frame_data.data.getvalue()

    from io import BytesIO

    from PIL import Image

    width, height = Image.open(BytesIO(image_bytes)).size
    encoded = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{frame_data.mime};base64,{encoded}"
    return data_url, width, height


def _build_content(
    *,
    user_prompt: str,
    label_names: Iterable[str],
    target_frame_id: int,
    target_image_data_url: str,
    context_image_data_urls: list[str],
) -> list[dict]:
    labels_prompt = ", ".join(sorted(label_names))
    instruction = (
        "You are assisting with image annotation.\n"
        f"Allowed classes for detections: [{labels_prompt}].\n"
        "Output rules:\n"
        "1) If the user asks for detections, return STRICT JSON only:\n"
        '{"detections":[{"class":"<label>","bbox":[x1,y1,x2,y2]}]}\n'
        "2) If the user asks for general advice/prompting and not detections, return plain text only.\n"
        "Detection constraints:\n"
        "- bbox values are pixel coordinates for the target image\n"
        "- x1 < x2 and y1 < y2\n"
        "- class must be from allowed classes\n"
        f"User prompt: {user_prompt}\n"
        f"Target frame id: {target_frame_id}."
    )

    content: list[dict] = [{"type": "input_text", "text": instruction}]

    for idx, data_url in enumerate(context_image_data_urls, start=1):
        content.append({"type": "input_text", "text": f"Context image {idx}"})
        content.append({"type": "input_image", "image_url": data_url})

    content.append({"type": "input_text", "text": "Target image"})
    content.append({"type": "input_image", "image_url": target_image_data_url})

    return content


def _call_openai(*, content: list[dict], model_name: str, timeout: int) -> tuple[str, dict | None]:
    openai_settings = settings.OPENAI
    api_key = openai_settings.get("API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI API key is not configured")

    endpoint = f"{openai_settings['URL'].rstrip('/')}/v1/responses"
    response = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_name,
            "input": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "temperature": 0.1,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()

    text = _extract_output_text(payload)
    if not text:
        raise ValueError("OpenAI returned an empty response")

    try:
        return text, _extract_json(text)
    except Exception:  # noqa: BLE001
        return text, None


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

    return [x1, y1, x2, y2]


def _has_any_annotations_on_frame(task_id: int, frame_id: int) -> bool:
    return (
        LabeledShape.objects.filter(job__segment__task_id=task_id, frame=frame_id).exists()
        or LabeledImage.objects.filter(job__segment__task_id=task_id, frame=frame_id).exists()
        or TrackedShape.objects.filter(track__job__segment__task_id=task_id, frame=frame_id).exists()
    )


def run_openai_annotation_task(
    task_id: int,
    frame_ids: list[int],
    context_frame_ids: list[int],
    prompt: str,
    options: dict,
) -> int:
    _LOG.info("[OPENAI] Starting annotation task=%s frames=%s", task_id, frame_ids)
    _set_current_job_state(message="Preparing task", progress=0.0)

    task = Task.objects.select_related("project", "data").get(pk=task_id)
    frame_provider = TaskFrameProvider(task)
    label_map = _label_mapping(task)

    target_frame_id = frame_ids[0]
    frame_provider.validate_frame_number(target_frame_id)

    timeout = max(1, int(options.get("timeout", settings.OPENAI.get("TIMEOUT", 120))))
    model = str(options.get("model", settings.OPENAI.get("MODEL", "gpt-4.1-mini")))

    context_image_data_urls: list[str] = []
    if context_frame_ids:
        context_frame_id = context_frame_ids[0]
        frame_provider.validate_frame_number(context_frame_id)
        if not _has_any_annotations_on_frame(task_id, context_frame_id):
            raise ValidationError("The selected context frame has no existing annotations.")

        context_url, _, _ = _to_image_data_url(frame_provider, context_frame_id)
        context_image_data_urls.append(context_url)

    _set_current_job_state(message="Calling ChatGPT", progress=0.25)
    target_url, width, height = _to_image_data_url(frame_provider, target_frame_id)
    content = _build_content(
        user_prompt=prompt,
        label_names=label_map.keys(),
        target_frame_id=target_frame_id,
        target_image_data_url=target_url,
        context_image_data_urls=context_image_data_urls,
    )

    # Exactly one OpenAI call per request
    raw_response_text, parsed_json = _call_openai(content=content, model_name=model, timeout=timeout)

    shapes: list[dict] = []
    if parsed_json is not None and "detections" in parsed_json:
        detections = parsed_json.get("detections")
        if not isinstance(detections, list):
            raise ValidationError("ChatGPT detections must be a list.")

        for idx, detection in enumerate(detections):
            if not isinstance(detection, dict):
                raise ValidationError(f"Detection #{idx + 1} is not an object.")

            raw_class = str(detection.get("class", "")).strip().lower()
            label_id = label_map.get(raw_class)
            if label_id is None:
                raise ValidationError(f"Detection #{idx + 1} has unknown class '{raw_class}'.")

            bbox = _normalize_bbox(detection.get("bbox", []), width, height)
            if bbox is None:
                raise ValidationError(f"Detection #{idx + 1} has invalid bbox.")

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
        _set_current_job_state(message=f"ChatGPT response: {text_message}", progress=1.0)
        _LOG.info("[OPENAI] Completed without detections for task=%s", task_id)
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
    _LOG.info("[OPENAI] Saved %s shapes for task=%s", len(shapes), task_id)
    return len(shapes)


@extend_schema(tags=["openai"])
class OpenAIViewSet(viewsets.GenericViewSet):
    permission_classes = [IsAuthenticated]
    serializer_class = OpenAIAnnotateSerializer
    iam_organization_field = None
    iam_permission_class = TaskPermission

    @extend_schema(
        methods=["POST"],
        summary="Generate annotations with ChatGPT",
        request=OpenAIAnnotateSerializer,
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

        manager = OpenAIAnnotateManager(
            request=request,
            db_instance=task,
            frame_ids=data["frame_ids"],
            context_frame_ids=data["context_frame_ids"],
            prompt=data["prompt"],
            options=data.get("options", {}),
        )
        res = manager.enqueue_job()
        return res
