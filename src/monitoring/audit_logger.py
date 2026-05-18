"""Audit logger — writes prediction records to DynamoDB for compliance and replay."""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from decimal import Decimal

import boto3

log = logging.getLogger(__name__)

_AUDIT_TABLE = os.environ.get("AUDIT_TABLE", "fraud-audit-log")
_OVERRIDE_TABLE = "fraud-override-queue"
_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")


def _to_decimal(value: float) -> Decimal:
    return Decimal(str(value))


def _from_dynamo(obj):
    """Recursively convert DynamoDB Decimal back to float."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _from_dynamo(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_dynamo(v) for v in obj]
    return obj


class AuditLogger:
    """Writes prediction records to DynamoDB for compliance and explainability."""

    def __init__(self) -> None:
        self._dynamo = boto3.resource("dynamodb", region_name=_REGION)
        self._audit_table = self._dynamo.Table(_AUDIT_TABLE)
        self._override_table = self._dynamo.Table(_OVERRIDE_TABLE)
        log.info("AuditLogger initialised  table=%s  region=%s", _AUDIT_TABLE, _REGION)

    async def write(
        self,
        prediction_id: str,
        input_features: dict,
        fraud_probability: float,
        is_fraud: bool,
        shap_values: dict,
        model_version: str,
        confidence_score: float,
        request_ip: str,
        latency_ms: float,
        threshold_used: float,
    ) -> None:
        """PutItem to DynamoDB. Fails silently to never block the prediction path."""
        item = {
            "prediction_id": prediction_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_features": json.dumps(input_features),
            "fraud_probability": _to_decimal(fraud_probability),
            "is_fraud": is_fraud,
            "shap_values": json.dumps(shap_values),
            "model_version": model_version,
            "confidence_score": _to_decimal(confidence_score),
            "request_ip": request_ip,
            "latency_ms": _to_decimal(latency_ms),
            "threshold_used": _to_decimal(threshold_used),
            # ttl omitted — audit log is permanent
        }
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self._audit_table.put_item(Item=item))
        except Exception as exc:
            log.error("AuditLogger.write failed (silent): %s", exc)

    async def fetch(self, prediction_id: str) -> dict | None:
        """GetItem by prediction_id. Returns None if not found or on error."""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._audit_table.get_item(Key={"prediction_id": prediction_id}),
            )
            item = response.get("Item")
            if item is None:
                return None
            record = _from_dynamo(item)
            for field in ("input_features", "shap_values"):
                if isinstance(record.get(field), str):
                    record[field] = json.loads(record[field])
            return record
        except Exception as exc:
            log.error("AuditLogger.fetch failed (silent): %s", exc)
            return None

    async def fetch_recent(self, limit: int = 500) -> list[dict]:
        """Scan the audit table and return up to limit records for drift computation."""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: self._audit_table.scan(Limit=limit))
            records: list[dict] = []
            for item in response.get("Items", []):
                record = _from_dynamo(item)
                for field in ("input_features", "shap_values"):
                    if isinstance(record.get(field), str):
                        record[field] = json.loads(record[field])
                records.append(record)
            return records
        except Exception as exc:
            log.error("AuditLogger.fetch_recent failed (silent): %s", exc)
            return []

    async def flag_override(
        self,
        prediction_id: str,
        reason: str,
        reviewer_notes: str | None = None,
    ) -> None:
        """PutItem to the override queue with a 30-day TTL."""
        item = {
            "prediction_id": prediction_id,
            "reason": reason or "",
            "reviewer_notes": reviewer_notes or "",
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "ttl": int(time.time()) + 30 * 24 * 3600,
        }
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self._override_table.put_item(Item=item))
            log.info("Override queued for prediction_id=%s", prediction_id)
        except Exception as exc:
            log.error("AuditLogger.flag_override failed (silent): %s", exc)
