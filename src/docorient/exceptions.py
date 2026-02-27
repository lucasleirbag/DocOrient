class DocorientError(Exception):
    pass


class DetectionError(DocorientError):
    pass


class CorrectionError(DocorientError):
    pass


class BatchProcessingError(DocorientError):
    pass
