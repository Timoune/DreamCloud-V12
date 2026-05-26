import requests

from config.model_config import (
    MAX_TOKENS,
    TEMPERATURE,
    TIMEOUT_SECONDS
)

from config.runtime_config import DEBUG


def run_llama(prompt):

    if DEBUG:
        print("\n[DEBUG] Sending request to llama-server...\n")

    try:
        response = requests.post(
            "http://127.0.0.1:8080/completion",
            json={
                "prompt": prompt,
                "n_predict": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "stop": ["User:", "user:", "Assistant:", "assistant:"]
            },
            timeout=TIMEOUT_SECONDS   # 🔥 FIXED
        )

        data = response.json()
        output = data.get("content", "")

        if DEBUG:
            print("\n[DEBUG OUTPUT PREVIEW]\n")
            print(output[:500])

        return output.strip()

    except Exception as e:
        return f"[ERROR] {str(e)}"