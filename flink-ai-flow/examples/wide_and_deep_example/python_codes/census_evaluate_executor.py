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

import tensorflow as tf
import ai_flow as af
from typing import List
from python_ai_flow import FunctionContext, Executor


class BatchEvaluateExecutor(Executor):
    def __init__(self):
        super().__init__()
        self.path = None
        self.model_version = None

    def setup(self, function_context: FunctionContext):
        model_name = function_context.node_spec.model.name
        listener_name = 'model_listener'
        # notifications = af.start_listen_notification(listener_name=listener_name, key=model_name)
        self.model_version = af.get_latest_generated_model_version(model_name)
        self.path = self.model_version.model_path
        # self._exported_model = self._model_path.split('|')[1]

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        x_evaluate, y_evaluate = input_list[0][0], input_list[0][1]
        x_evaluate = x_evaluate / 255.0
        model = tf.keras.models.load_model(self.path)
        result = model.evaluate(x_evaluate, y_evaluate, verbose=0)
        output = function_context.node_spec.output_result
        path = output.stream_uri
        with open(path, 'a') as f:
            f.write(str(result) + '  -------->  ' + self.model_version.version)
            f.write('\n')
        return []
