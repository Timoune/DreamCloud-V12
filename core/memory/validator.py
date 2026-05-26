def validate_memory(data):
 
    required = [
        "id",
        "timestamp",
        "content"
    ]
 
    for key in required:
 
        if key not in data:
            return False
 
    return True