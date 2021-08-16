# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from abc import ABC, abstractmethod
from typing import Text

from notification_service.base_notification import BaseEvent


class ContextExtractor(ABC):
    """
    ContextExtractor is used to decide if a event should be broadcast or extract context from a event.
    If the event should be broadcast, it will be handle by all the workflow execution and job execution
    of that workflow. Otherwise, only workflow execution and job execution with the same context can handle the event.
    """

    @abstractmethod
    def extract_context(self, event: BaseEvent) -> Text:
        """
        If the event is not to be broadcast, this method is called to extract the context from the event. The event will
        only be handled by the workflow execution and job execution under the same context. If the None is returned,
        workflow execution and job execution under default context will handle the event.

        :param event: The event to extract context from.
        :return: The context of the event.
        """
        pass

    @abstractmethod
    def is_broadcast_event(self, event: BaseEvent) -> bool:
        """
        Decide if the event should be broadcast. If True, the event will be handled by all the workflow execution and
        job execution in the workflow. Otherwise, extract_context will be called to decide the context of the event.

        :param event: The event to check if it should be broadcast.
        :return: Whether the event should be broadcast.
        """
        pass
