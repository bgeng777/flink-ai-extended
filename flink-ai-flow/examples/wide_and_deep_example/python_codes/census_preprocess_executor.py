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

from typing import List
from python_ai_flow import FunctionContext, Executor
import pandas as pd
from sklearn.utils import shuffle


class BatchPreprocessExecutor(Executor):

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        data_path = '/tmp/census_data/adult.data'
        df = pd.read_csv(data_path, header=None)
        df = shuffle(df)
        df.to_csv('/tmp/census_data/adult.data', index=False, header=None)
        return []
