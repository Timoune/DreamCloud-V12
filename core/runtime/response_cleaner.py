def clean_response(text):

    stop_markers = [
        "\nUser:",
        "\nuser:",
        "\nAssistant:",
        "\nassistant:"
    ]

    # 🔥 hard cut at role markers
    for marker in stop_markers:
        if marker in text:
            text = text.split(marker)[0]

    lines = text.splitlines()

    cleaned = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        lower = stripped.lower()

        # 🔥 remove role lines + instruction echo
        if lower.startswith(("user:", "assistant:")):
            continue

        if "answer only" in lower:
            continue

        cleaned.append(stripped)

    return "\n".join(cleaned).strip()