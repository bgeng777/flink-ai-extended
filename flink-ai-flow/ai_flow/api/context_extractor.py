from abc import abstractmethod
from typing import Text, Union
from ai_flow.util.json_utils import Jsonable
from notification_service.base_notification import BaseEvent


class ContextExtractor(Jsonable):
    """
    ContextExtractor is written by users to decide how to extract the context of a workflow execution
    (or job executions in that workflow execution) from the event subscribed by the workflow or the job.
    It is a workflow-level config.
    """

    @abstractmethod
    def extract_context(self, event: BaseEvent) -> Union[Text, None]:
        """
        Extract context(string) from the given event

        :param event: The :class:`~notification_service.base_notification.BaseEvent` that should be processed.
        :return: The context that this event belongs to.
        :rtype: Text
        """
        pass


class DefaultContextExtractor(ContextExtractor):

    def extract_context(self, event: BaseEvent) -> Union[Text, None]:
        return None
