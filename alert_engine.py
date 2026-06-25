def fire_alert(fire_count):
    if fire_count >= 1:
        return {
            "alert": True,
            "type": "fire",
            "message": " Fire detected!",
            "count": fire_count,
            "severity": "high"
        }
    else:
        return {
            "alert": False,
            "type": "fire",
            "message": "No fire detected",
            "count": fire_count,
            "severity": "none"
        }


def accident_alert(accident_count):
    if accident_count >= 1:
        return {
            "alert": True,
            "type": "accident",
            "message": " Accident detected!",
            "count": accident_count,
            "severity": "high"
        }
    else:
        return {
            "alert": False,
            "type": "accident",
            "message": "No accident detected",
            "count": accident_count,
            "severity": "none"
        }


def sos_alert(label):
    emergency_labels = [
        "حريقه",
        "حادثه",
        "سرقه",
        "اعتداء",
        "استغاثه",
        "عادي"
    ]

    if label.lower() in emergency_labels:
        return {
            "alert": True,
            "type": "sos",
            "message": f" SOS detected: {label}",
            "severity": "critical"
        }
    else:
        return {
            "alert": False,
            "type": "sos",
            "message": f"Normal message: {label}",
            "severity": "low"
        }