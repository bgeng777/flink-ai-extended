from typing import Text, Set
from notification_service.base_notification import BaseEvent

from ai_flow.api.context_extractor import ContextExtractor, EventContext


class TestContext2(EventContext):
    """
    This class indicates that the event shouxld be broadcast.
    """

    def is_broadcast(self) -> bool:
        return True

    def get_contexts(self) -> Set[Text]:
        s = set()
        s.add('hello2')
        return s

class TestContextExtractor2(ContextExtractor):
    """
    BroadcastAllContextExtractor is the default ContextExtractor to used. It marks all events as broadcast events.
    """

    def extract_context(self, event: BaseEvent) -> EventContext:
        return TestContext2()

