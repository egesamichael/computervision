from __future__ import annotations

import csv
from io import StringIO

from flask import Blueprint, Response, current_app, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from .recommendations import build_recommendations
from .storage import (
    count_predictions,
    delete_all_predictions,
    delete_prediction,
    list_predictions,
    list_recent_predictions,
    save_prediction,
)

main = Blueprint("main", __name__)


@main.route("/", methods=["GET", "POST"])
def index():
    context = {
        "error": None,
        "recent_predictions": [],
        "sample_type": "leaf",
        "location": "",
        "notes": "",
        "tm_model_url": current_app.config.get("TM_MODEL_URL"),
        "tm_metadata_url": current_app.config.get("TM_METADATA_URL"),
        "store_predictions": current_app.config.get("STORE_PREDICTIONS"),
        "top_predictions": current_app.config.get("TOP_PREDICTIONS", 5),
    }

    if request.method == "POST":
        context["error"] = (
            "Browser-based inference is required. Please enable JavaScript."
        )

    context["recent_predictions"] = _load_recent_predictions()
    return render_template("index.html", **context)


def _load_recent_predictions() -> list[dict[str, str | float]]:
    if not current_app.config.get("STORE_PREDICTIONS"):
        return []
    try:
        return list_recent_predictions(
            current_app.config["PREDICTIONS_DB"],
            limit=current_app.config["RECENT_PREDICTIONS_LIMIT"],
        )
    except Exception:
        current_app.logger.exception("Failed to load recent prediction history.")
        return []


@main.route("/history", methods=["GET"])
def history():
    context = {
        "error": request.args.get("error") or None,
        "predictions": [],
        "total": 0,
        "limit": 50,
    }

    if not current_app.config.get("STORE_PREDICTIONS"):
        context["error"] = "Prediction storage is disabled."
        return render_template("history.html", **context)

    try:
        limit = min(max(int(request.args.get("limit", "50")), 1), 200)
    except ValueError:
        limit = 50
    context["limit"] = limit

    try:
        context["predictions"] = list_predictions(
            current_app.config["PREDICTIONS_DB"], limit=limit
        )
        context["total"] = count_predictions(current_app.config["PREDICTIONS_DB"])
    except Exception:
        current_app.logger.exception("Failed to load prediction history page.")
        context["error"] = "Unable to load prediction history right now."

    return render_template("history.html", **context)


@main.route("/history.csv", methods=["GET"])
def history_csv():
    if not current_app.config.get("STORE_PREDICTIONS"):
        return Response("Prediction storage is disabled.", status=400)

    try:
        limit = min(max(int(request.args.get("limit", "500")), 1), 2000)
    except ValueError:
        limit = 500

    try:
        rows = list_predictions(
            current_app.config["PREDICTIONS_DB"], limit=limit
        )
    except Exception:
        current_app.logger.exception("Failed to export prediction history.")
        return Response("Unable to export prediction history.", status=500)

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "created_at",
            "filename",
            "sample_type",
            "location",
            "notes",
            "top_label",
            "top_confidence",
        ]
    )
    for row in rows:
        writer.writerow(
            [
                row.get("created_at", ""),
                row.get("filename", ""),
                row.get("sample_type", ""),
                row.get("location", ""),
                row.get("notes", ""),
                row.get("top_label", ""),
                f'{row.get("top_confidence", 0):.4f}',
            ]
        )

    response = Response(output.getvalue(), mimetype="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=prediction_history.csv"
    return response


@main.route("/api/diagnose", methods=["POST"])
def diagnose_api():
    payload = request.get_json(silent=True) or {}
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        return {"error": "Invalid predictions payload."}, 400

    cleaned: list[dict[str, float | str]] = []
    for item in predictions:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        if not label:
            continue
        try:
            confidence = float(item.get("confidence", 0))
        except (TypeError, ValueError):
            continue
        confidence = max(0.0, min(1.0, confidence))
        cleaned.append({"label": label, "confidence": confidence})

    if not cleaned:
        return {"error": "No usable predictions provided."}, 400

    cleaned.sort(key=lambda item: float(item["confidence"]), reverse=True)
    top_prediction = cleaned[0]

    sample_type = str(payload.get("sample_type", "leaf")).strip().lower()
    if sample_type not in {"leaf", "berry", "other"}:
        sample_type = "leaf"

    recommendations = build_recommendations(
        sample_type=sample_type,
        top_label=str(top_prediction["label"]),
        confidence=float(top_prediction["confidence"]),
    )

    filename = str(payload.get("filename") or "upload")
    mime_type = str(payload.get("mime_type") or "image/jpeg")
    location = payload.get("location") or None
    notes = payload.get("notes") or None

    if current_app.config.get("STORE_PREDICTIONS"):
        try:
            save_prediction(
                current_app.config["PREDICTIONS_DB"],
                filename=secure_filename(filename) or "upload",
                mime_type=mime_type,
                sample_type=sample_type,
                location=location,
                notes=notes,
                top_label=str(top_prediction["label"]),
                top_confidence=float(top_prediction["confidence"]),
                predictions=cleaned,
            )
        except Exception:
            current_app.logger.exception("Failed to save prediction history.")
            return {"error": "Failed to save prediction history."}, 500

    return {
        "ok": True,
        "diagnosis": {
            "label": str(top_prediction["label"]),
            "confidence": float(top_prediction["confidence"]),
        },
        "recommendations": recommendations,
    }


@main.route("/history/delete/<int:prediction_id>", methods=["POST"])
def delete_history_item(prediction_id: int):
    if not current_app.config.get("STORE_PREDICTIONS"):
        return Response("Prediction storage is disabled.", status=400)
    try:
        delete_prediction(current_app.config["PREDICTIONS_DB"], prediction_id)
    except Exception:
        current_app.logger.exception(
            "Failed to delete prediction history entry %s", prediction_id
        )
        return redirect(
            url_for("main.history", error="Unable to delete that entry right now.")
        )
    return redirect(url_for("main.history"))


@main.route("/history/clear", methods=["POST"])
def clear_history():
    if not current_app.config.get("STORE_PREDICTIONS"):
        return Response("Prediction storage is disabled.", status=400)
    try:
        delete_all_predictions(current_app.config["PREDICTIONS_DB"])
    except Exception:
        current_app.logger.exception("Failed to clear prediction history.")
        return redirect(
            url_for("main.history", error="Unable to clear history right now.")
        )
    return redirect(url_for("main.history"))
