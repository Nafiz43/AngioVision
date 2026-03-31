import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# ---------------------------------------------------------------------
# Claude Opus 4.6
# WORKING COMBINATION from your experiment:
#
#   REGION   = us-east-1
#   MODEL_ID = global.anthropic.claude-opus-4-6-v1
#
# Do NOT use:
#   anthropic.claude-opus-4-6-v1
# because direct on-demand invocation is not supported.
# ---------------------------------------------------------------------

MODEL_ID = "global.anthropic.claude-opus-4-6-v1"
REGION = "us-east-1"

IMAGE_PATH = "/data/Deep_Angiography/DICOM_Sequence_Processed/0AVNTO~C/2.16.840.1.113883.3.16.242948424383568667903940832500591782968/mosaic.png"
IMAGE_FORMAT = "png"

QUESTION = "Which artery is visible?"
MAX_TOKENS = 512
TEMPERATURE = 0


def main():
    try:
        print(f"[INFO] Using region: {REGION}")
        print(f"[INFO] Using modelId: {MODEL_ID}")

        session = boto3.Session()
        credentials = session.get_credentials()

        if credentials is None:
            raise NoCredentialsError()

        frozen_creds = credentials.get_frozen_credentials()
        print("[INFO] AWS credentials found.")
        print(f"[INFO] Access key starts with: {frozen_creds.access_key[:4]}****")

        client = session.client("bedrock-runtime", region_name=REGION)

        if not os.path.exists(IMAGE_PATH):
            raise FileNotFoundError(IMAGE_PATH)

        with open(IMAGE_PATH, "rb") as f:
            image_bytes = f.read()

        response = client.converse(
            modelId=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "image": {
                                "format": IMAGE_FORMAT,
                                "source": {
                                    "bytes": image_bytes
                                }
                            }
                        },
                        {
                            "text": QUESTION
                        }
                    ]
                }
            ],
            inferenceConfig={
                "maxTokens": MAX_TOKENS,
                "temperature": TEMPERATURE
            }
        )

        content = response.get("output", {}).get("message", {}).get("content", [])
        answer_parts = [item["text"] for item in content if "text" in item]
        answer = "\n".join(answer_parts).strip()

        print("\n[Claude response]\n")
        print(answer if answer else "[No text returned]")

    except FileNotFoundError as e:
        print(f"[ERROR] Image file not found: {e}")

    except NoCredentialsError:
        print("[ERROR] AWS credentials not found.")
        print("Make sure your AWS credentials are exported in the same shell session,")
        print("or configured via ~/.aws/credentials, AWS_PROFILE, or an IAM role.")

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        error_message = e.response.get("Error", {}).get("Message", str(e))

        print(f"[AWS Client Error] {error_code}: {error_message}")

        if error_code == "AccessDeniedException":
            print("\n[HINT]")
            print("This usually means model/profile access is still blocked for this account.")
        elif error_code == "ValidationException":
            print("\n[HINT]")
            print("Check that REGION is us-east-1 and MODEL_ID is exactly:")
            print("global.anthropic.claude-opus-4-6-v1")

    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")


if __name__ == "__main__":
    main()