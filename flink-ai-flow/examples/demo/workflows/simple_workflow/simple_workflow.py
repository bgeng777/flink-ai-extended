from typing import Optional, Text

import cloudpickle
from ai_flow.api.context_extractor import ContextExtractor
from notification_service.base_notification import BaseEvent

import ai_flow as af
from ai_flow_plugins.job_plugins.bash import BashProcessor


def main():
    af.init_ai_flow_context()
    with af.job_config('task_1'):
        af.user_define_operation(processor=BashProcessor("echo hello"))
    with af.job_config('task_2'):
        af.user_define_operation(processor=BashProcessor("echo hello"))

    af.action_on_job_status('task_2', 'task_1')

    workflow_name = af.current_workflow_config().workflow_name
    # af.set_context_extractor(MyContextExtractor)
    # stop_workflow_executions(workflow_name)
    af.workflow_operation.submit_workflow(workflow_name)
    # af.workflow_operation.start_new_workflow_execution(workflow_name)
    print("----")
    t = af.get_ai_flow_client().get_workflow_by_name('demo7', workflow_name)
    print("----")
    print(t.context_extractor_in_bytes)
    cc = cloudpickle.loads(t.context_extractor_in_bytes.value)
    a=cc()
    print(dir(cc))
    print(a.extract_context(BaseEvent(key='1', value='1')))


def stop_workflow_executions(workflow_name):
    workflow_executions = af.workflow_operation.list_workflow_executions(workflow_name)
    for workflow_execution in workflow_executions:
        af.workflow_operation.stop_workflow_execution(workflow_execution.workflow_execution_id)


if __name__ == '__main__':
    main()
