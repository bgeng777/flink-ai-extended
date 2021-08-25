#
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
#
import os
import time

from typing import List

from notification_service.base_notification import BaseEvent, ANY_CONDITION

from ai_flow_plugins.job_plugins.python.python_processor import PythonProcessor, ExecutionContext

import ai_flow as af
from ai_flow.util.path_util import get_file_dir
from ai_flow.model_center.entity.model_version_stage import ModelVersionEventType
import shutil

DATASET_URI = os.path.abspath(os.path.join(__file__, "../../../")) + '/resources/iris_train.csv'


class DatasetMaker(PythonProcessor):
    def __init__(self, raw_dataset, project_name):
        super().__init__()
        self.raw_dataset = raw_dataset
        self.project_name = project_name

    def process(self, execution_context: ExecutionContext, input_list: List) -> List:
        """
        Read dataset using pandas
        """
        i = 0
        while i < 4:
            timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
            new_dataset_path = self.raw_dataset[:-4] + timestamp + '.csv'
            shutil.copy(self.raw_dataset, new_dataset_path)
            af.get_ai_flow_client().send_event(BaseEvent(event_type='DATA_EVENT', key='hourly_data', value='ready',
                                                         context=timestamp, namespace=self.project_name,
                                                         sender=ANY_CONDITION))
            time.sleep(20)
            i+=1
        return []


def run_workflow():
    # Init project
    af.init_ai_flow_context()
    print(af.current_project_config().get_project_name())
    # Training of model
    with af.job_config('data_gen'):
        af.user_define_operation(processor=DatasetMaker(DATASET_URI, af.current_project_config().get_project_name()))
    workflow_name = af.current_workflow_config().workflow_name
    # Submit workflow
    stop_workflow_executions(workflow_name)
    af.workflow_operation.submit_workflow(workflow_name)
    # Run workflow
    af.workflow_operation.start_new_workflow_execution(workflow_name)


def stop_workflow_executions(workflow_name):
    workflow_executions = af.workflow_operation.list_workflow_executions(workflow_name)
    for workflow_execution in workflow_executions:
        af.workflow_operation.stop_workflow_execution(workflow_execution.workflow_execution_id)


if __name__ == '__main__':
    run_workflow()
