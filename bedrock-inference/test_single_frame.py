import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

REGION = "us-west-2"

IMAGE_PATH = "/data/Deep_Angiography/DICOM_Sequence_Processed/0AVNTO~C/2.16.840.1.113883.3.16.242948424383568667903940832500591782968/mosaic.png"
IMAGE_FORMAT = "jpeg"


def main():
    try:
        client = boto3.client("bedrock-runtime", region_name=REGION)

        # Load image
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
                            "text": "Which artery is visible??"
                        }
                    ]
                }
            ],
            inferenceConfig={
                "maxTokens": 512,
                "temperature": 0.2
            }
        )

        answer = response["output"]["message"]["content"][0]["text"]

        print("\nClaude response:\n")
        print(answer)

    except FileNotFoundError:
        print("Image file not found:", IMAGE_PATH)

    except NoCredentialsError:
        print("AWS credentials not found.")

    except ClientError as e:
        print("AWS client error:", e)

    except Exception as e:
        print("Unexpected error:", e)


if __name__ == "__main__":
    main()