class ProteinAgentError(Exception):
    """Base exception for expected service failures."""


class ProteinNotFoundError(ProteinAgentError):
    pass


class UpstreamServiceError(ProteinAgentError):
    def __init__(self, service: str, message: str) -> None:
        super().__init__(f"{service}: {message}")
        self.service = service


class InvalidProteinRequestError(ProteinAgentError):
    pass
