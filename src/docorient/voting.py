from __future__ import annotations

from collections import Counter

from docorient.types import OrientationResult


def apply_majority_voting(
    detection_results: list[OrientationResult],
) -> list[OrientationResult]:
    confident_angles = [
        result.angle for result in detection_results if result.reliable
    ]

    if not confident_angles:
        return detection_results

    majority_angle = Counter(confident_angles).most_common(1)[0][0]
    corrected_results = []

    for result in detection_results:
        if not result.reliable and result.angle != majority_angle:
            corrected_results.append(
                OrientationResult(
                    angle=majority_angle,
                    method=f"{result.method}->majority({majority_angle},was={result.angle})",
                    reliable=True,
                )
            )
        else:
            corrected_results.append(result)

    return corrected_results
