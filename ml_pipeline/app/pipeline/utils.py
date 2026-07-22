def format_time(seconds):
    """
    Convert seconds to HH:MM:SS
    """

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    return f"{hours:02}:{minutes:02}:{secs:02}"


def get_cardinal_direction(yaw):
    """
    Convert yaw degrees (0-360) to a cardinal direction (N, NE, E, etc.)
    """
    if yaw is None:
        return None
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((yaw + 22.5) % 360 / 45)
    return directions[idx]