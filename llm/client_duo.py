from llm.llm_client import BaseLLMClient

class ClientDuo:
    def __init__(self, boardroomClient: BaseLLMClient, summaryClient: BaseLLMClient = None):
        self.boardroomClient = boardroomClient
        self.summaryClient = summaryClient

        if summaryClient is None:
            self.summaryClient = self.boardroomClient