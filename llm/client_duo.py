from llm.llm_client import BaseLLMClient

class ClientDuo:
    def __init__(self, boardroomClient: BaseLLMClient, summaryClient: BaseLLMClient):
        self.boardroomClient = boardroomClient
        self.summaryClient = summaryClient