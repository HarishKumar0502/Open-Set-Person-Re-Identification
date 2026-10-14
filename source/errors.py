class PS1Error(Exception):
    pass


class ConfigError(PS1Error):
    pass


class ArtifactError(PS1Error):
    pass


class CheckpointError(ArtifactError):
    pass


class ImageError(PS1Error):
    pass


class DetectorUnavailableError(PS1Error):
    pass


class BackendError(PS1Error):
    pass


class NoBackendAvailable(BackendError):
    pass
