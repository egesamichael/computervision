from __future__ import annotations

from base64 import b64encode

import csv
from io import StringIO

from flask import Blueprint, Response, current_app, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from .model import classifier
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


def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in current_app.config["ALLOWED_EXTENSIONS"]
    )


@main.route("/", methods=["GET", "POST"])
def index():
    context = {
        "error": None,
        "filename": None,
        "image_data": None,
        "model_name": classifier.default_name,
        "model_notice": None,
        "predictions": [],
        "diagnosis": None,
        "recommendations": [],
        "recent_predictions": [],
        "sample_type": "leaf",
        "location": "",
        "notes": "",
    }

    if request.method == "POST":
        sample_type = request.form.get("sample_type", "leaf").strip().lower()
        if sample_type not in {"leaf", "berry", "other"}:
            sample_type = "leaf"
        context["sample_type"] = sample_type
        context["location"] = request.form.get("location", "").strip()
        context["notes"] = request.form.get("notes", "").strip()

        uploaded_file = request.files.get("image")
        if uploaded_file is None or uploaded_file.filename == "":
            context["error"] = "Choose an image before running prediction."
            context["recent_predictions"] = _load_recent_predictions()
            return render_template("index.html", **context)

        filename = secure_filename(uploaded_file.filename)
        context["filename"] = filename

        if not allowed_file(filename):
            allowed = ", ".join(sorted(current_app.config["ALLOWED_EXTENSIONS"]))
            context["error"] = f"Unsupported file type. Use one of: {allowed}."
            context["recent_predictions"] = _load_recent_predictions()
            return render_template("index.html", **context)

        raw_bytes = uploaded_file.read()
        if not raw_bytes:
            context["error"] = "The uploaded file is empty."
            context["recent_predictions"] = _load_recent_predictions()
            return render_template("index.html", **context)

        try:
            context["predictions"] = classifier.predict(
                raw_bytes,
                top=current_app.config["TOP_PREDICTIONS"],
            )
            mime_type = uploaded_file.mimetype or "image/jpeg"
            encoded_image = b64encode(raw_bytes).decode("utf-8")
            context["image_data"] = f"data:{mime_type};base64,{encoded_image}"
            context["model_name"] = classifier.name
            context["model_notice"] = classifier.notice

            if context["predictions"]:
                top_prediction = context["predictions"][0]
                context["diagnosis"] = {
                    "label": top_prediction["label"],
                    "confidence": top_prediction["confidence"],
                }
                context["recommendations"] = build_recommendations(
                    sample_type=sample_type,
                    top_label=top_prediction["label"],
                    confidence=top_prediction["confidence"],
                )

                if current_app.config.get("STORE_PREDICTIONS"):
                    try:
                        save_prediction(
                            current_app.config["PREDICTIONS_DB"],
                            filename=filename,
                            mime_type=mime_type,
                            sample_type=sample_type,
                            location=context["location"] or None,
                            notes=context["notes"] or None,
                            top_label=top_prediction["label"],
                            top_confidence=top_prediction["confidence"],
                            predictions=context["predictions"],
                        )
                    except Exception:
                        current_app.logger.exception(
                            "Failed to save prediction history for %s", filename
                        )
        except Exception:
            current_app.logger.exception("Prediction failed for %s", filename)
            context["error"] = (
                "Prediction failed. Upload a valid image and try again."
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
