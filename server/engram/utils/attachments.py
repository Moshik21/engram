"""Attachment helpers for multimodal embedding pipeline."""

from __future__ import annotations

import base64
import logging

logger = logging.getLogger(__name__)


def decode_attachment(data_url: str) -> bytes:
    """Decode base64 data from a data URI or raw base64 string.

    Supports two formats:
      - ``data:image/png;base64,iVBOR...`` (RFC 2397 data URI)
      - Raw base64 string (e.g. ``iVBOR...``)
    """
    if data_url.startswith("data:"):
        _, encoded = data_url.split(",", 1)
        return base64.b64decode(encoded)
    return base64.b64decode(data_url)


def get_first_image_attachment(
    attachments: list,
) -> tuple[bytes, str] | None:
    """Return decoded (image_bytes, mime_type) for the first image attachment.

    Returns ``None`` if no image attachment is found.  Only considers
    attachments whose ``mime_type`` starts with ``image/``.
    """
    for att in attachments:
        if att.mime_type.startswith("image/"):
            try:
                image_bytes = decode_attachment(att.data_url)
                return image_bytes, att.mime_type
            except Exception:
                logger.warning(
                    "Failed to decode image attachment (mime=%s), skipping",
                    att.mime_type,
                )
    return None
