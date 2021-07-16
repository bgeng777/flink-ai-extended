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
from enum import Enum
from typing import Text

from ai_flow.util.json_utils import Jsonable
from notification_service.base_notification import UNDEFINED_EVENT_TYPE, DEFAULT_NAMESPACE

from ai_flow.graph.edge import Edge


class ConditionType(str, Enum):
    """
    SUFFICIENT: Sufficient conditions, as long as one condition is met, the action will be triggered.
    NECESSARY: Necessary conditions, all conditions are met before the action is triggered.
    """
    SUFFICIENT = "SUFFICIENT"
    """
    SUFFICIENT: Sufficient conditions, as long as one condition is met, the action will be triggered.
    """
    NECESSARY = "NECESSARY"


class TaskAction(str, Enum):
    """START: Start the job.

    RESTART: If the job is running, stop it first and then start it. If the job is not running,just start it.

    STOP: Stop the job.

    NONE: Do nothing.

    """
    START = "START"
    RESTART = "RESTART"
    STOP = "STOP"
    NONE = "NONE"


class EventLife(str, Enum):
    """
    ONCE: The event value will be used only once.
    REPEATED: The event value will be used repeated.
    """
    ONCE = "ONCE"
    REPEATED = "REPEATED"


class ValueCondition(str, Enum):
    """
    EQUALS: The condition that notification service updates a value which equals to the event value.
    UPDATED: The condition that notification service has a update operation on the event key which event
            value belongs.
    """
    EQUALS = "EQUALS"
    UPDATED = "UPDATE"


class ConditionConfig(Jsonable):
    def __init__(self,
                 event_key: Text,
                 event_value: Text,
                 event_type: Text = UNDEFINED_EVENT_TYPE,
                 namespace: Text = DEFAULT_NAMESPACE,
                 sender: Text = None,
                 condition_type: ConditionType = ConditionType.NECESSARY,
                 action: TaskAction = TaskAction.START,
                 life: EventLife = EventLife.ONCE,
                 value_condition: ValueCondition = ValueCondition.EQUALS
                 ):
        """
        :param event_key: The Key of the event(notification_service.base_notification.BaseEvent).
        :param event_value: The value of the event(notification_service.base_notification.BaseEvent).
        :param namespace: The namespace of the event(notification_service.base_notification.BaseEvent).
        :param event_type: (Optional) Type of the event(notification_service.base_notification.BaseEvent).
        :param sender: The sender of the event(notification_service.base_notification.BaseEvent).
        :param condition_type: ai_flow.workflow.control_edge.ConditionType
        :param action: ai_flow.workflow.control_edge.TaskAction
        :param life: ai_flow.workflow.control_edge.EventLife
        :param value_condition: ai_flow.workflow.control_edge.MetValueCondition
        """
        self.event_type = event_type
        self.event_key = event_key
        self.event_value = event_value
        self.condition_type = condition_type
        self.action = action
        self.life = life
        self.value_condition = value_condition
        self.namespace = namespace
        self.sender = sender


class ControlEdge(Edge):
    """
    ControlEdge defines event-based dependencies between jobs(ai_flow.workflow.job.Job).
    """
    def __init__(self,
                 destination: Text,
                 condition_config: ConditionConfig,
                 ) -> None:
        """
        :param destination: The name of the job which depends on condition_config.
        :param condition_config: ai_flow.workflow.control_edge.ConditionConfig
        """
        super().__init__(condition_config.sender, destination)
        self.condition_config = condition_config


class AIFlowInternalEventType(object):
    """Per-defined some event types which are generated by ai flow system."""
    JOB_STATUS_CHANGED = "JOB_STATUS_CHANGED"  # Indicates the job(ai_flow.workflow.job.Job) status changed event.
    PERIODIC_ACTION = "PERIODIC_ACTION"  # Indicates the type of event that a job or workflow runs periodically.
    DATASET_CHANGED = "DATASET_CHANGED"  # Indicates the type of dataset changed event.
