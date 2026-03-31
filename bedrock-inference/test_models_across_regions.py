import os
import re
import csv
import json
import time
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# ============================================================
# CONFIG
# ============================================================

REGIONS = ["us-west-2", "us-east-1"]

# Main probing image/question
IMAGE_PATH = "/data/Deep_Angiography/DICOM_Sequence_Processed/0AVNTO~C/2.16.840.1.113883.3.16.242948424383568667903940832500591782968/mosaic.png"
IMAGE_FORMAT = "png"
QUESTION = "Which artery is visible?"

MAX_TOKENS = 256
TEMPERATURE = 0.2

# Filtering
ONLY_IMAGE_CAPABLE = True
ONLY_CLAUDE_4X = True  # If True, tests Claude 4 / 4.1 / 4.5 / 4.6 families
INCLUDE_HAIKU_45 = True

# Explicit known profile mappings
# Add more here if you later discover them from the console.
EXPLICIT_INFERENCE_PROFILES = {
    "anthropic.claude-sonnet-4-6": {
        "profile_id": "global.anthropic.claude-sonnet-4-6",
        "profile_arn": "arn:aws:bedrock:us-west-2:944446239581:inference-profile/global.anthropic.claude-sonnet-4-6",
    }
}

# Optional AngioVision batch input:
# CSV with columns:
#   AccessionNumber,SOPInstanceUID,mosaic_path,Question
ANGIOVISION_BATCH_CSV = None
# Example:
# ANGIOVISION_BATCH_CSV = "/data/Deep_Angiography/AngioVision/bedrock-inference/angio_batch.csv"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.getcwd()

LOG_TXT = os.path.join(OUT_DIR, f"bedrock_claude4_probe_{TIMESTAMP}.log")
LOG_JSON = os.path.join(OUT_DIR, f"bedrock_claude4_probe_{TIMESTAMP}.json")
LOG_CSV = os.path.join(OUT_DIR, f"bedrock_claude4_probe_{TIMESTAMP}.csv")

ANGIOVISION_OUT_CSV = os.path.join(OUT_DIR, f"angiovision_bedrock_predictions_{TIMESTAMP}.csv")


# ============================================================
# HELPERS
# ============================================================

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_line(msg: str, fh=None):
    line = f"[{now_str()}] {msg}"
    print(line)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def safe_read_image_bytes(path: str) -> bytes:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    with open(path, "rb") as f:
        return f.read()


def extract_text_from_converse(resp: dict) -> str:
    try:
        content = resp.get("output", {}).get("message", {}).get("content", [])
        texts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
        return "\n".join(texts).strip()
    except Exception:
        return ""


def supports_image(model_summary: dict) -> bool:
    modalities = model_summary.get("inputModalities", []) or []
    return "IMAGE" in [m.upper() for m in modalities]


def is_claude_model(model_summary: dict) -> bool:
    model_id = (model_summary.get("modelId") or "").lower()
    model_name = (model_summary.get("modelName") or "").lower()
    provider_name = (model_summary.get("providerName") or "").lower()
    return (
        "anthropic" in provider_name
        or "claude" in model_id
        or "claude" in model_name
    )


def is_claude_4x_model(model_summary: dict) -> bool:
    model_id = (model_summary.get("modelId") or "").lower()

    # Include:
    # - Claude Sonnet 4
    # - Claude Sonnet 4.5
    # - Claude Sonnet 4.6
    # - Claude Opus 4
    # - Claude Opus 4.1
    # - Claude Opus 4.5
    # - Claude Opus 4.6
    # - Optional: Haiku 4.5
    patterns = [
        r"anthropic\.claude-sonnet-4(?:-|$)",
        r"anthropic\.claude-opus-4(?:-|$)",
    ]
    if INCLUDE_HAIKU_45:
        patterns.append(r"anthropic\.claude-haiku-4-5(?:-|$)")

    return any(re.search(p, model_id) for p in patterns)


def dedupe_models(models: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for m in models:
        mid = m.get("modelId")
        if not mid or mid in seen:
            continue
        seen.add(mid)
        out.append(m)
    return out


def sanitize_filename_component(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def get_account_id(session: boto3.Session) -> Optional[str]:
    try:
        sts = session.client("sts")
        return sts.get_caller_identity()["Account"]
    except Exception:
        return None


# ============================================================
# SEMI-HEURISTIC INFERENCE PROFILE DISCOVERY
# ============================================================

def build_profile_candidates_for_model(
    model_id: str,
    account_id: Optional[str],
    active_region: str,
) -> List[Tuple[str, str]]:
    """
    Returns list of (attempt_type, candidate_id_or_arn).
    This is heuristic, not guaranteed exhaustive.
    """
    candidates: List[Tuple[str, str]] = []
    seen = set()

    def add(kind: str, value: str):
        if value and value not in seen:
            seen.add(value)
            candidates.append((kind, value))

    # 1) Direct model first
    add("direct_model", model_id)

    # 2) Explicit known mapping
    if model_id in EXPLICIT_INFERENCE_PROFILES:
        p = EXPLICIT_INFERENCE_PROFILES[model_id]
        if "profile_id" in p:
            add("explicit_profile_id", p["profile_id"])
        if "profile_arn" in p:
            add("explicit_profile_arn", p["profile_arn"])

    # 3) Heuristic profile ID
    # Example:
    # anthropic.claude-sonnet-4-6 -> global.anthropic.claude-sonnet-4-6
    if model_id.startswith("anthropic."):
        heuristic_profile_id = f"global.{model_id}"
        add("heuristic_profile_id", heuristic_profile_id)

        # 4) Heuristic ARNs across target regions
        if account_id:
            for r in ["us-west-2", "us-east-1", active_region]:
                heuristic_arn = f"arn:aws:bedrock:{r}:{account_id}:inference-profile/{heuristic_profile_id}"
                add("heuristic_profile_arn", heuristic_arn)

    return candidates


# ============================================================
# MODEL ENUMERATION
# ============================================================

def get_candidate_models(session: boto3.Session, region: str, log_fh=None) -> List[dict]:
    bedrock = session.client("bedrock", region_name=region)
    resp = bedrock.list_foundation_models()
    all_models = resp.get("modelSummaries", [])

    log_line(f"{region}: total listed foundation models = {len(all_models)}", log_fh)

    models = [m for m in all_models if is_claude_model(m)]
    models = dedupe_models(models)

    if ONLY_IMAGE_CAPABLE:
        models = [m for m in models if supports_image(m)]

    if ONLY_CLAUDE_4X:
        models = [m for m in models if is_claude_4x_model(m)]

    models = sorted(models, key=lambda x: (x.get("modelName") or "", x.get("modelId") or ""))

    log_line(f"{region}: candidate models after filtering = {len(models)}", log_fh)
    for idx, m in enumerate(models, start=1):
        log_line(
            f"  {idx}. {m.get('modelName')} | {m.get('modelId')} | inputModalities={m.get('inputModalities')}",
            log_fh
        )

    return models


# ============================================================
# SINGLE ATTEMPT
# ============================================================

def try_converse(
    runtime_client,
    model_ref: str,
    image_bytes: bytes,
    question: str,
    max_tokens: int,
    temperature: float,
) -> dict:
    return runtime_client.converse(
        modelId=model_ref,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "image": {
                            "format": IMAGE_FORMAT,
                            "source": {"bytes": image_bytes}
                        }
                    },
                    {"text": question}
                ]
            }
        ],
        inferenceConfig={
            "maxTokens": max_tokens,
            "temperature": temperature
        }
    )


# ============================================================
# PROBE ONE MODEL
# ============================================================

def probe_model(
    session: boto3.Session,
    region: str,
    model: dict,
    image_bytes: bytes,
    question: str,
    account_id: Optional[str],
    global_seq: int,
    regional_seq: int,
    log_fh=None
) -> dict:
    runtime = session.client("bedrock-runtime", region_name=region)

    model_id = model.get("modelId")
    model_name = model.get("modelName", model_id)
    input_modalities = model.get("inputModalities", [])

    result = {
        "globalSequence": global_seq,
        "regionSequence": regional_seq,
        "region": region,
        "modelName": model_name,
        "modelId": model_id,
        "inputModalities": input_modalities,
        "worked": False,
        "winningAttemptType": None,
        "winningModelRef": None,
        "latencySeconds": None,
        "responsePreview": None,
        "errorType": None,
        "errorMessage": None,
        "attempts": [],
        "timestamp": now_str(),
    }

    attempts = build_profile_candidates_for_model(model_id, account_id, region)

    log_line("-" * 100, log_fh)
    log_line(f"GLOBAL #{global_seq} | REGION #{regional_seq} | {region}", log_fh)
    log_line(f"Model name: {model_name}", log_fh)
    log_line(f"Model id:   {model_id}", log_fh)
    log_line(f"Attempts planned: {len(attempts)}", log_fh)

    for attempt_idx, (attempt_type, model_ref) in enumerate(attempts, start=1):
        log_line(f"Attempt {attempt_idx}: {attempt_type} -> {model_ref}", log_fh)
        started = time.time()

        attempt_record = {
            "attemptIndex": attempt_idx,
            "attemptType": attempt_type,
            "modelRef": model_ref,
            "worked": False,
            "latencySeconds": None,
            "errorType": None,
            "errorMessage": None,
        }

        try:
            resp = try_converse(
                runtime_client=runtime,
                model_ref=model_ref,
                image_bytes=image_bytes,
                question=question,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            elapsed = round(time.time() - started, 3)
            text = extract_text_from_converse(resp)

            attempt_record["worked"] = True
            attempt_record["latencySeconds"] = elapsed
            result["attempts"].append(attempt_record)

            result["worked"] = True
            result["winningAttemptType"] = attempt_type
            result["winningModelRef"] = model_ref
            result["latencySeconds"] = elapsed
            result["responsePreview"] = text[:500] if text else ""

            log_line(f"SUCCESS via {attempt_type} | latency={elapsed}s", log_fh)
            if text:
                log_line(f"Preview: {text[:200]}", log_fh)

            return result

        except ClientError as e:
            elapsed = round(time.time() - started, 3)
            error_type = e.response.get("Error", {}).get("Code", "ClientError")
            error_msg = e.response.get("Error", {}).get("Message", str(e))

            attempt_record["latencySeconds"] = elapsed
            attempt_record["errorType"] = error_type
            attempt_record["errorMessage"] = error_msg
            result["attempts"].append(attempt_record)

            log_line(f"FAILED via {attempt_type} | {error_type}: {error_msg}", log_fh)

        except Exception as e:
            elapsed = round(time.time() - started, 3)
            error_type = type(e).__name__
            error_msg = str(e)

            attempt_record["latencySeconds"] = elapsed
            attempt_record["errorType"] = error_type
            attempt_record["errorMessage"] = error_msg
            result["attempts"].append(attempt_record)

            log_line(f"FAILED via {attempt_type} | {error_type}: {error_msg}", log_fh)

    # If all attempts failed, keep the last error on the top-level result
    if result["attempts"]:
        last = result["attempts"][-1]
        result["errorType"] = last.get("errorType")
        result["errorMessage"] = last.get("errorMessage")

    return result


# ============================================================
# MULTI-REGION PROBE
# ============================================================

def run_multiregion_probe() -> List[dict]:
    session = boto3.Session()

    creds = session.get_credentials()
    if creds is None:
        raise NoCredentialsError()

    account_id = get_account_id(session)
    image_bytes = safe_read_image_bytes(IMAGE_PATH)

    all_results: List[dict] = []
    global_seq = 1

    with open(LOG_TXT, "w", encoding="utf-8") as log_f:
        log_line("Starting Claude 4.x multi-region probe", log_f)
        log_line(f"Regions: {REGIONS}", log_f)
        log_line(f"Image path: {IMAGE_PATH}", log_f)
        log_line(f"Question: {QUESTION}", log_f)
        log_line(f"Account ID detected: {account_id}", log_f)
        log_line(f"ONLY_IMAGE_CAPABLE={ONLY_IMAGE_CAPABLE}", log_f)
        log_line(f"ONLY_CLAUDE_4X={ONLY_CLAUDE_4X}", log_f)

        for region in REGIONS:
            log_line("=" * 100, log_f)
            log_line(f"BEGIN REGION {region}", log_f)

            try:
                models = get_candidate_models(session, region, log_f)
            except Exception as e:
                log_line(f"Could not enumerate models in {region}: {e}", log_f)
                continue

            regional_seq = 1
            for model in models:
                res = probe_model(
                    session=session,
                    region=region,
                    model=model,
                    image_bytes=image_bytes,
                    question=QUESTION,
                    account_id=account_id,
                    global_seq=global_seq,
                    regional_seq=regional_seq,
                    log_fh=log_f,
                )
                all_results.append(res)
                global_seq += 1
                regional_seq += 1

        log_line("=" * 100, log_f)
        log_line("FINAL SUMMARY", log_f)

        worked = [r for r in all_results if r["worked"]]
        failed = [r for r in all_results if not r["worked"]]

        log_line(f"Total tested: {len(all_results)}", log_f)
        log_line(f"Worked: {len(worked)}", log_f)
        log_line(f"Failed: {len(failed)}", log_f)

        if worked:
            log_line("WORKED:", log_f)
            for r in worked:
                log_line(
                    f"  region={r['region']} | model={r['modelId']} | winner={r['winningAttemptType']} | ref={r['winningModelRef']}",
                    log_f
                )

        if failed:
            log_line("FAILED:", log_f)
            for r in failed:
                log_line(
                    f"  region={r['region']} | model={r['modelId']} | last_error={r['errorType']} | {r['errorMessage']}",
                    log_f
                )

    with open(LOG_JSON, "w", encoding="utf-8") as jf:
        json.dump(all_results, jf, indent=2)

    write_probe_csv(all_results, LOG_CSV)
    return all_results


# ============================================================
# CLEAN CSV EXPORT
# ============================================================

def write_probe_csv(results: List[dict], csv_path: str):
    fieldnames = [
        "globalSequence",
        "regionSequence",
        "region",
        "modelName",
        "modelId",
        "worked",
        "winningAttemptType",
        "winningModelRef",
        "latencySeconds",
        "errorType",
        "errorMessage",
        "responsePreview",
        "timestamp",
        "numAttempts",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            writer.writerow({
                "globalSequence": r.get("globalSequence"),
                "regionSequence": r.get("regionSequence"),
                "region": r.get("region"),
                "modelName": r.get("modelName"),
                "modelId": r.get("modelId"),
                "worked": r.get("worked"),
                "winningAttemptType": r.get("winningAttemptType"),
                "winningModelRef": r.get("winningModelRef"),
                "latencySeconds": r.get("latencySeconds"),
                "errorType": r.get("errorType"),
                "errorMessage": r.get("errorMessage"),
                "responsePreview": r.get("responsePreview"),
                "timestamp": r.get("timestamp"),
                "numAttempts": len(r.get("attempts", [])),
            })


# ============================================================
# ANGIOVISION INTEGRATION
# ============================================================

def choose_best_successful_model(results: List[dict]) -> Optional[dict]:
    """
    Picks a successful model result, preferring Sonnet 4.6 if it works.
    Otherwise picks the first successful result.
    """
    successful = [r for r in results if r.get("worked")]
    if not successful:
        return None

    preferred_patterns = [
        "anthropic.claude-sonnet-4-6",
        "anthropic.claude-opus-4-6-v1",
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "anthropic.claude-sonnet-4-20250514-v1:0",
        "anthropic.claude-haiku-4-5-20251001-v1:0",
    ]

    for p in preferred_patterns:
        for r in successful:
            if r.get("modelId") == p:
                return r

    return successful[0]


def ask_bedrock_with_selected_ref(
    session: boto3.Session,
    region: str,
    model_ref: str,
    image_path: str,
    question: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> str:
    runtime = session.client("bedrock-runtime", region_name=region)
    image_bytes = safe_read_image_bytes(image_path)

    resp = runtime.converse(
        modelId=model_ref,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "image": {
                            "format": IMAGE_FORMAT,
                            "source": {"bytes": image_bytes}
                        }
                    },
                    {"text": question}
                ]
            }
        ],
        inferenceConfig={
            "maxTokens": max_tokens,
            "temperature": temperature
        }
    )
    return extract_text_from_converse(resp)


def run_angiovision_batch(
    batch_csv_path: str,
    selected_region: str,
    selected_model_ref: str,
    output_csv_path: str,
):
    """
    Expected input columns:
      AccessionNumber,SOPInstanceUID,mosaic_path,Question
    """
    if not os.path.exists(batch_csv_path):
        raise FileNotFoundError(f"AngioVision batch CSV not found: {batch_csv_path}")

    session = boto3.Session()
    creds = session.get_credentials()
    if creds is None:
        raise NoCredentialsError()

    with open(batch_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    fieldnames = [
        "Timestamp",
        "Region",
        "ModelRef",
        "AccessionNumber",
        "SOPInstanceUID",
        "Question",
        "Predicted",
        "RawOutput",
        "mosaic_path",
        "status",
        "error",
    ]

    with open(output_csv_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(rows, start=1):
            accession = row.get("AccessionNumber", "")
            sopuid = row.get("SOPInstanceUID", "")
            mosaic_path = row.get("mosaic_path", "")
            question = row.get("Question", "").strip()

            if not question:
                question = QUESTION

            status = "success"
            error = ""
            raw_output = ""
            predicted = ""

            try:
                raw_output = ask_bedrock_with_selected_ref(
                    session=session,
                    region=selected_region,
                    model_ref=selected_model_ref,
                    image_path=mosaic_path,
                    question=question,
                    max_tokens=256,
                    temperature=0.0,
                )
                predicted = raw_output.strip()

            except Exception as e:
                status = "failed"
                error = str(e)

            writer.writerow({
                "Timestamp": now_str(),
                "Region": selected_region,
                "ModelRef": selected_model_ref,
                "AccessionNumber": accession,
                "SOPInstanceUID": sopuid,
                "Question": question,
                "Predicted": predicted,
                "RawOutput": raw_output,
                "mosaic_path": mosaic_path,
                "status": status,
                "error": error,
            })

            print(f"[{idx}/{len(rows)}] {status} | {accession} | {sopuid}")


# ============================================================
# MAIN
# ============================================================

def main():
    try:
        results = run_multiregion_probe()

        print("\nProbe finished.")
        print(f"Text log:  {LOG_TXT}")
        print(f"JSON log:  {LOG_JSON}")
        print(f"CSV log:   {LOG_CSV}")

        best = choose_best_successful_model(results)
        if best:
            print("\nBest successful model reference selected for downstream use:")
            print(f"  Region:    {best['region']}")
            print(f"  Model ID:  {best['modelId']}")
            print(f"  Use ref:   {best['winningModelRef']}")
            print(f"  Attempt:   {best['winningAttemptType']}")
        else:
            print("\nNo successful model was found.")
            return

        # Optional AngioVision batch inference
        if ANGIOVISION_BATCH_CSV:
            print("\nRunning AngioVision batch...")
            run_angiovision_batch(
                batch_csv_path=ANGIOVISION_BATCH_CSV,
                selected_region=best["region"],
                selected_model_ref=best["winningModelRef"],
                output_csv_path=ANGIOVISION_OUT_CSV,
            )
            print(f"AngioVision output CSV: {ANGIOVISION_OUT_CSV}")

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
    except NoCredentialsError:
        print("[ERROR] AWS credentials not found.")
        print("Make sure they are exported in the same shell, or configured via ~/.aws/credentials / AWS_PROFILE / IAM role.")
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()