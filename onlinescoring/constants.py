from enum import Enum

class ServerSetupParams:
    """Parameters for setting up the server."""

    WAIT_TIME_MIN = 15  # time to wait for the server to become healthy
    DEFAULT_WORKER_COUNT = 1

class WebServer:
    HOST = "localhost"
    PORT = 8000

class TaskType(str, Enum):
    """Enum representing the types of tasks."""

    TEXT_GENERATION = "text-generation"
    CONVERSATIONAL = "conversational"
    TEXT_TO_IMAGE = "text-to-image"
    TEXT_CLASSIFICATION = "text-classification"
    TEXT_CLASSIFICATION_MULTILABEL = "text-classification-multilabel"
    NER = "text-named-entity-recognition"
    SUMMARIZATION = "text-summarization"
    QnA = "question-answering"
    TRANSLATION = "text-translation"
    TEXT_GENERATION_CODE = "text-generation-code"
    FILL_MASK = "fill-mask"
    CHAT_COMPLETION = "chat-completion"
    TEXT_TO_IMAGE_INPAINTING = "text-to-image-inpainting"

class SupportedTask:
    """Supported tasks by text-generation-inference."""

    TEXT_GENERATION = "text-generation"
    CHAT_COMPLETION = "chat-completion"
    TEXT_TO_IMAGE = "text-to-image"
    TEXT_CLASSIFICATION = "text-classification"
    TEXT_CLASSIFICATION_MULTILABEL = "text-classification-multilabel"
    NER = "token-classification"
    SUMMARIZATION = "summarization"
    QnA = "question-answering"
    TRANSLATION = "translation"
    TEXT_GENERATION_CODE = "text-generation-code"
    FILL_MASK = "fill-mask"
    TEXT_TO_IMAGE_INPAINTING = "text-to-image-inpainting"

ALL_TASKS = [
    SupportedTask.TEXT_TO_IMAGE,
    SupportedTask.TEXT_CLASSIFICATION,
    SupportedTask.TEXT_CLASSIFICATION_MULTILABEL,
    SupportedTask.NER,
    SupportedTask.SUMMARIZATION,
    SupportedTask.QnA,
    SupportedTask.TRANSLATION,
    SupportedTask.FILL_MASK,
    SupportedTask.TEXT_GENERATION,
    SupportedTask.CHAT_COMPLETION,
]