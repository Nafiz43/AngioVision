"""All HTTP routes for the AngioVision query web server (registered as a Blueprint)."""

import json
import time
import logging
from typing import Any, Dict, Optional

from flask import Blueprint, request, jsonify, Response, render_template

from . import config
from .deps import IMAGE_DEPS_OK, SMOLAGENTS_OK, base64
from .state import state
from .db import get_db_stats, run_sql_query, open_db
from .agent import run_nl_query_agent
from .embeddings import get_embedding_model
from .images import (
    get_chroma_collection,
    decode_image_to_uint8_rgb,
    enrich_results_from_sqlite,
    load_dicom_frame_as_b64,
)
from .llm import set_model, llm_call
from .prompts import IMAGE_SYNTHESIS_SYSTEM

log = logging.getLogger(__name__)

bp = Blueprint("api", __name__)


@bp.route("/api/stats", methods=["GET"])
def api_stats():
    """GET /api/stats — database statistics and report linkage coverage."""
    try:
        return jsonify(get_db_stats())
    except Exception as exc:
        log.exception("GET /api/stats failed")
        return jsonify({"error": str(exc)}), 500


def _completed_sequences_for_model(model_key: str) -> Optional[int]:
    """
    Count completed image-ingestion sequences for a given embedding model.

    Returns None if the count can't be determined. On databases predating
    per-model tracking (no embedding_model column) the count is only meaningful
    for the default model, so it is reported only in that case (else 0).
    """
    try:
        with state.lock:
            con = open_db()
            try:
                cols = [r[1] for r in con.execute(
                    "PRAGMA table_info(image_ingestion_status)"
                ).fetchall()]
                if not cols:
                    return None
                if "embedding_model" in cols:
                    return con.execute(
                        "SELECT COUNT(*) FROM image_ingestion_status "
                        "WHERE status='completed' AND embedding_model=?",
                        (model_key,),
                    ).fetchone()[0]
                # Legacy schema: existing rows are implicitly the default model.
                if model_key == config.DEFAULT_EMBEDDING_MODEL:
                    return con.execute(
                        "SELECT COUNT(*) FROM image_ingestion_status WHERE status='completed'"
                    ).fetchone()[0]
                return 0
            finally:
                con.close()
    except Exception as exc:
        log.debug(f"Could not fetch sequence count: {exc}")
        return None


@bp.route("/api/embedding-models", methods=["GET"])
def api_embedding_models():
    """GET /api/embedding-models — registry of selectable image-embedding models."""
    models = [
        {
            "key":        key,
            "label":      spec["label"],
            "hf_id":      spec["hf_id"],
            "collection": spec["collection"],
        }
        for key, spec in config.EMBEDDING_MODELS.items()
    ]
    return jsonify({"models": models, "default": config.DEFAULT_EMBEDDING_MODEL})


@bp.route("/api/chroma-stats", methods=["GET"])
def api_chroma_stats():
    """
    GET /api/chroma-stats?model=<key> — ChromaDB statistics for the selected
    embedding model's collection (defaults to the default model).
    """
    if not IMAGE_DEPS_OK:
        return jsonify({
            "available": False,
            "error": "Image dependencies not installed (chromadb, Pillow, numpy)",
        })

    model_key = config.resolve_embedding_model(request.args.get("model"))
    spec      = config.EMBEDDING_MODELS[model_key]
    try:
        col = get_chroma_collection(model_key)
        if col is None:
            return jsonify({
                "available":   False,
                "count":       0,
                "sequences":   0,
                "model":       model_key,
                "model_label": spec["label"],
                "hf_id":       spec["hf_id"],
                "collection":  spec["collection"],
            })

        return jsonify({
            "available":   True,
            "count":       col.count(),
            "sequences":   _completed_sequences_for_model(model_key),
            "model":       model_key,
            "model_label": spec["label"],
            "hf_id":       spec["hf_id"],
            "collection":  spec["collection"],
            "path":        str(state.chromadb_path),
        })
    except Exception as exc:
        log.exception("GET /api/chroma-stats failed")
        return jsonify({
            "available":   False,
            "error":       str(exc),
            "model":       model_key,
            "model_label": spec["label"],
        })


@bp.route("/api/query", methods=["POST"])
def api_query() -> Response:
    """
    POST /api/query — Agentic NL→SQL query resolution via smolagents ToolCallingAgent.

    Request body: {"question": "...", "think": true/false, "model": "qwen3:1.7b"}
    Returns a Server-Sent Events (SSE) stream (events match the original pipeline).
    """
    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    think = data.get("think", state.think)
    model = (data.get("model") or config.DEFAULT_MODEL).strip()

    if not question:
        log.warning("POST /api/query: missing 'question' parameter")
        return jsonify({"error": "question required"}), 400

    if not SMOLAGENTS_OK:
        log.error("POST /api/query: smolagents not installed")
        return jsonify({
            "error": "smolagents is not installed on the server. Run: pip install 'smolagents[openai]'"
        }), 503

    def generate():
        t0 = time.time()

        def emit(obj: Dict[str, Any]) -> str:
            return "data: " + json.dumps(obj) + "\n\n"

        yield emit({"event": "sql_start"})

        had_error = False
        for evt in run_nl_query_agent(question, think, model):
            yield emit(evt)
            if evt.get("event") == "error":
                had_error = True
                break

        if not had_error:
            elapsed = int((time.time() - t0) * 1000)
            yield emit({"event": "done", "elapsed_ms": elapsed})

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@bp.route("/api/image-query", methods=["POST"])
def api_image_query() -> Response:
    """
    POST /api/image-query — embedding-model + ChromaDB visual similarity search.

    Body: {"image": "<base64>", "question": "...", "n_results": 5, "think": true,
           "embedding_model": "rad-dino"}
    The query image is embedded with the selected model and matched against that
    model's own ChromaDB collection. Streams SSE events; groups frames by
    SOPInstanceUID so each result represents a full series.
    """
    data      = request.get_json(force=True)
    b64_image = data.get("image", "").strip()
    question  = data.get("question", "Show me the most similar cases to this image.").strip()
    n_results = max(1, min(20, int(data.get("n_results", config.N_SIMILAR_DEFAULT))))
    think     = data.get("think", state.think)
    model_key   = config.resolve_embedding_model(data.get("embedding_model"))
    model_label = config.EMBEDDING_MODELS[model_key]["label"]

    if not b64_image:
        return jsonify({"error": "image required"}), 400

    if not IMAGE_DEPS_OK:
        return jsonify({
            "error": (
                "Image query dependencies not installed on server. "
                "Run: pip install chromadb pillow numpy torch torchvision timm"
            )
        }), 503

    def generate():
        t0 = time.time()

        def emit(obj: dict) -> str:
            return "data: " + json.dumps(obj) + "\n\n"

        # ── Step 1: Decode image ─────────────────────────────────────────────
        yield emit({"event": "decode_start"})
        try:
            img_array = decode_image_to_uint8_rgb(b64_image)
            h, w      = img_array.shape[:2]
            yield emit({"event": "decode_done", "width": int(w), "height": int(h)})
        except Exception as exc:
            yield emit({"event": "error", "message": f"Image decode failed: {exc}"})
            return

        # ── Step 2: ChromaDB similarity search ───────────────────────────────
        yield emit({"event": "chroma_start"})
        try:
            col = get_chroma_collection(model_key)
            if col is None:
                yield emit({
                    "event":   "error",
                    "message": (
                        f"ChromaDB collection for '{model_label}' not available. "
                        f"Build it first: python3 run_ingest.py --images-only "
                        f"--embedding-model {model_key}"
                    ),
                })
                return

            available = col.count()
            if available == 0:
                yield emit({
                    "event":   "error",
                    "message": (
                        f"No vectors indexed for '{model_label}' yet. Build its "
                        f"collection first: python3 run_ingest.py --images-only "
                        f"--embedding-model {model_key}"
                    ),
                })
                return

            # Over-fetch to allow grouping by SOPInstanceUID
            fetch_n = min(available, config.N_SIMILAR_OVERFETCH)

            # Embed the query with the selected model
            ef = get_embedding_model(model_key)
            if ef is None:
                # Fallback to query_images parameter
                raw = col.query(
                    query_images=[img_array],
                    n_results=fetch_n,
                    include=["metadatas", "distances"],
                )
            else:
                # Use the selected model's embeddings
                query_embedding = ef([img_array])[0]
                raw = col.query(
                    query_embeddings=[query_embedding],
                    n_results=fetch_n,
                    include=["metadatas", "distances"],
                )

            ids       = raw["ids"][0]
            distances = raw["distances"][0]
            metadatas = raw["metadatas"][0]

            # Group by accession_number → then by sop_uid within each accession
            acc_groups: dict[str, dict] = {}
            for id_, dist, meta in zip(ids, distances, metadatas):
                acc = meta.get("accession_number") or "UNKNOWN"
                sop = meta.get("sop_uid", "") or "UNKNOWN"
                sim = max(0.0, 1.0 - float(dist))
                frame_entry = {
                    "chroma_id":      id_,
                    "similarity_pct": round(sim * 100, 1),
                    "distance":       round(float(dist), 4),
                    "source_path":    meta.get("source_path", ""),
                    "frame_index":    meta.get("frame_index", 0),
                    "sop_uid":        sop,
                }
                if acc not in acc_groups:
                    acc_groups[acc] = {
                        "accession_number": acc,
                        "_top_sim":         sim,
                        "sop_groups":       {},   # { sop_uid: {frames, _top_sim} }
                        **{k: (v if v is not None else "") for k, v in meta.items()},
                    }
                if sim > acc_groups[acc]["_top_sim"]:
                    acc_groups[acc]["_top_sim"] = sim

                # Inner group: one entry per sop_uid
                sop_dict = acc_groups[acc]["sop_groups"]
                if sop not in sop_dict:
                    sop_dict[sop] = {"sop_uid": sop, "_top_sim": sim, "frames": []}
                if sim > sop_dict[sop]["_top_sim"]:
                    sop_dict[sop]["_top_sim"] = sim
                sop_dict[sop]["frames"].append(frame_entry)

            # Finalise: sort frames within each SOP, sort SOPs within each accession
            for grp in acc_groups.values():
                sop_list = []
                for sg in grp["sop_groups"].values():
                    sg["frames"].sort(key=lambda f: f["similarity_pct"], reverse=True)
                    sg["similarity_pct"] = round(sg.pop("_top_sim") * 100, 1)
                    sop_list.append(sg)
                sop_list.sort(key=lambda s: s["similarity_pct"], reverse=True)
                grp["sop_groups"] = sop_list
                grp["similarity_pct"] = round(grp["_top_sim"] * 100, 1)

            # Sort accessions by top similarity, keep top n_results
            top = sorted(acc_groups.values(), key=lambda x: x["_top_sim"], reverse=True)[:n_results]
            results = []
            for i, r in enumerate(top):
                r.pop("_top_sim", None)
                r["rank"] = i + 1
                results.append(r)

            yield emit({"event": "chroma_done", "results": results, "count": len(results)})

        except Exception as exc:
            yield emit({"event": "error", "message": f"ChromaDB query failed: {exc}"})
            return

        # ── Step 3: Enrich from SQLite ───────────────────────────────────────
        yield emit({"event": "enrich_start"})
        try:
            enriched = enrich_results_from_sqlite(results)
        except Exception as exc:
            log.warning(f"Enrichment error (non-fatal): {exc}")
            enriched = results
        yield emit({"event": "enrich_done", "results": enriched})

        # ── Step 4: LLM synthesis ────────────────────────────────────────────
        yield emit({"event": "synth_start"})
        try:
            cases_text = "\n".join(
                f"#{r['rank']} ({len(r.get('sop_groups', []))} SOP(s), top similarity {r['similarity_pct']}%): "
                f"Accession={r.get('accession_number','?')}, "
                f"Date={r.get('study_date','?')}, "
                f"Modality={r.get('modality','?')}, "
                f"Series={r.get('series_description','?')}, "
                f"Age={r.get('patient_age','?')}, Sex={r.get('patient_sex','?')}"
                + (f"\n   Report excerpt: {str(r['radrpt_excerpt'])[:250]}"
                   if r.get("radrpt_excerpt") else "")
                for r in enriched
            )
            answer = llm_call([
                {"role": "system", "content": IMAGE_SYNTHESIS_SYSTEM},
                {"role": "user",   "content": (
                    f"Question: {question}\n\n"
                    f"Top {len(enriched)} visually similar series retrieved from ChromaDB "
                    f"({model_label} embedding similarity):\n\n{cases_text}"
                )},
            ], think=False)
        except Exception as exc:
            answer = f"(Synthesis unavailable: {exc})"

        elapsed = int((time.time() - t0) * 1000)
        yield emit({"event": "answer",  "text": answer})
        yield emit({"event": "done",    "elapsed_ms": elapsed})

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@bp.route("/api/thumbnail", methods=["GET"])
def api_thumbnail() -> Response:
    """GET /api/thumbnail?path=<file_path>&frame=<frame_index> — 320px PNG thumbnail."""
    path = request.args.get("path", "").strip()
    if not path:
        log.warning("GET /api/thumbnail: missing 'path' parameter")
        return jsonify({"error": "path required"}), 400

    try:
        frame_index = int(request.args.get("frame", 0))
    except ValueError:
        frame_index = 0

    thumbnail_b64 = load_dicom_frame_as_b64(path, frame_index, thumb_px=320)
    if thumbnail_b64:
        img_bytes = base64.b64decode(thumbnail_b64)
        return Response(
            img_bytes,
            mimetype="image/png",
            headers={"Cache-Control": "public, max-age=3600"},
        )
    else:
        log.debug(f"GET /api/thumbnail: frame not extracted [{path}:{frame_index}]")
        return Response(status=404)


@bp.route("/api/frame", methods=["GET"])
def api_frame() -> Response:
    """GET /api/frame?path=<file_path>&frame=<frame_index> — full-resolution PNG (lightbox)."""
    path = request.args.get("path", "").strip()
    if not path:
        log.warning("GET /api/frame: missing 'path' parameter")
        return jsonify({"error": "path required"}), 400

    try:
        frame_index = int(request.args.get("frame", 0))
    except ValueError:
        frame_index = 0

    frame_b64 = load_dicom_frame_as_b64(path, frame_index, thumb_px=2048)
    if frame_b64:
        img_bytes = base64.b64decode(frame_b64)
        return Response(
            img_bytes,
            mimetype="image/png",
            headers={"Cache-Control": "public, max-age=3600"},
        )
    else:
        log.debug(f"GET /api/frame: frame not extracted [{path}:{frame_index}]")
        return Response(status=404)


@bp.route("/api/model", methods=["POST"])
def api_set_model():
    """POST /api/model — switch the Ollama model used for synthesis."""
    data = request.get_json(force=True)
    model = data.get("model", "").strip()
    if not model:
        log.warning("POST /api/model: missing 'model' parameter")
        return jsonify({"error": "model required"}), 400
    set_model(model)
    log.info(f"POST /api/model: switched to {model}")
    return jsonify({"ok": True, "model": model})


@bp.route("/api/sql", methods=["POST"])
def api_run_sql():
    """POST /api/sql — execute arbitrary read-only SQL (advanced / debugging use)."""
    data = request.get_json(force=True)
    sql = data.get("sql", "").strip()
    if not sql:
        log.warning("POST /api/sql: missing 'sql' parameter")
        return jsonify({"error": "sql required"}), 400
    try:
        rows = run_sql_query(sql)

        def serialize_value(v: Any) -> Any:
            if v is None:
                return None
            try:
                json.dumps(v)
                return v
            except Exception:
                return str(v)

        clean_rows = [{k: serialize_value(v) for k, v in row.items()} for row in rows]
        return jsonify({"rows": clean_rows, "row_count": len(clean_rows)})
    except Exception as exc:
        log.exception("POST /api/sql: execution failed")
        return jsonify({"error": str(exc)}), 400


@bp.route("/guide", methods=["GET"])
def guide() -> str:
    """Serve the How-to-use guide page."""
    return render_template("guide.html")


@bp.route("/about", methods=["GET"])
def about() -> str:
    """Serve the About / team page."""
    return render_template("about.html")


@bp.route("/", methods=["GET"])
def index() -> str:
    """Serve the main HTML UI for the DICOM query engine."""
    return render_template("index.html")
