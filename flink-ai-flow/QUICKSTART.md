# Quickstart

The Quickstart will help you get started with an example in AI Flow. 
To begin with, we provide 2 ways for users to use AI Flow: 1) Install AI Flow on the local machine; 2) Run a docker image.

## Install on Local Machine
### Prerequisites
1. python3.7
2. pip

### Install AI Flow

If you are installing AI Flow from source, you can install AI Flow by running the following command:

```shell
# remove the build cache if exists
cd flink-ai-extended
rm -rf ai_flow.egg-info
rm -rf build
rm -rf dist
#python3 setup.py bdist_wheel
#python3 -m pip install dist/*.whl
pip uninstall typing
pip install ./flink-ai-flow/lib/notification_service
sudo pip install ./flink-ai-flow/lib/airflow
pip install ./flink-ai-flow/
```

If you are installing AI Flow from the release package, just run:

```shell
python3 -m pip install ai_flow-xxx-none-any.whl
```

If you meet any problems during the installation, please refer to the [Troubleshooting](#troubleshooting) section to see if it can help.

### Run a Python AI Flow Example

Here is a simple AI Flow example, and a line-by-line explanation follows right below. 

```python
import tempfile
import textwrap

from typing import List

from airflow.logging_config import configure_logging

import ai_flow
from python_ai_flow import Executor


def create_server_config(root_dir_path):
    content = textwrap.dedent(f"""\
        # Config of master server
        
        # endpoint of master
        master_ip: localhost
        master_port: 50051
        # uri of database backend in master
        db_uri: sqlite:///{root_dir_path}/aiflow.db
        # type of database backend in master
        db_type: sql_lite
        # the default notification service should be enabled 
        # when using the built-in scheduler, 
        start_default_notification: True
        # uri of the default notification service
        notification_uri: localhost:50052
    """)
    master_yaml_path = root_dir_path + "/master.yaml"
    with open(master_yaml_path, "w") as f:
        f.write(content)
    return master_yaml_path


def create_project_config(root_dir_path):
    content = textwrap.dedent("""\
        # Config of project in client

        # name of the project
        project_name: test_project
        # endpoint of master
        master_ip: localhost
        master_port: 50051
    """)
    project_yaml_path = root_dir_path + "/project.yaml"
    with open(project_yaml_path, "w") as f:
        f.write(content)
    return project_yaml_path


def create_workflow_config(root_dir_path):
    content = textwrap.dedent("""\
        # Config of the jobs
        
        # Config of each job
        job_1:
            # Use local platform
            platform: local
            # Operation type is python
            engine: python
            # Name of the job
            job_name: job_1
        job_2:
            platform: local
            engine: python
            job_name: job_2
    """)
    workflow_config_path = root_dir_path + "/workflow_config.yaml"
    with open(root_dir_path + "/workflow_config.yaml", "w") as f:
        f.write(content)
    return workflow_config_path


def start_master(master_yaml_path):
    # enable the logging
    configure_logging()
    # create and start the AI Flow master server
    master = ai_flow.AIFlowMaster(config_file=master_yaml_path)
    master.start(is_block=False)
    return master


class PrintHelloExecutor(Executor):
    """
    A simple executor which just print "hello world!".
    """

    def __init__(self, job_name):
        super().__init__()
        self.job_name = job_name

    def execute(self, function_context: ai_flow.FunctionContext, input_list: List) -> None:
        print("hello world! {}".format(self.job_name))


def build_workflow(workflow_config_path):
    with ai_flow.global_config_file(workflow_config_path):
        # a workflow contains one or more jobs
        with ai_flow.config('job_1'):
            # a job contains one or mode operations
            op_1 = ai_flow.user_define_operation(
                ai_flow.PythonObjectExecutor(PrintHelloExecutor('job_1')))

        with ai_flow.config('job_2'):
            op_2 = ai_flow.user_define_operation(
                ai_flow.PythonObjectExecutor(PrintHelloExecutor('job_2')))

        # start op_2 after op_1 finished
        ai_flow.stop_before_control_dependency(op_2, op_1)


def run_workflow(root_dir_path, project_yaml_path):
    # set the project config file
    ai_flow.set_project_config_file(project_yaml_path)
    # use the built-in scheduler ai_flow.SchedulerType.AIFLOW
    res = ai_flow.run(root_dir_path,
                      dag_id='hello_world_example',
                      scheduler_type=ai_flow.SchedulerType.AIFLOW)
    # wait until the workflow finished
    ai_flow.wait_workflow_execution_finished(res)


if __name__ == '__main__':
    # all files will be placed at a temporary directory
    root_dir = tempfile.mkdtemp()

    # create the master server config
    master_yaml = create_server_config(root_dir)
    # create the project config
    project_yaml = create_project_config(root_dir)
    # create the workflow config
    workflow_config = create_workflow_config(root_dir)

    # start a master server. 
    # for simplify we start it in current process, 
    # normally it should be started as a standalone server.
    master_server = start_master(master_yaml)

    # build the workflow
    build_workflow(workflow_config)
    # run the workflow
    run_workflow(root_dir, project_yaml)

    # as we started the master server in current process, we need to stop it at the end.
    master_server.stop()
    # the outputs of the python jobs can be found at the "${root_dir}/logs" directory.
    print("The output could be found in: %s/logs/" % root_dir)

```

You can run it with following command in your terminal.:
```shell
python examples/simple_examples/python_codes/hello_world_example.py
```

The output in the logs directory should be:

*1_job_1_{timestamp}_stdout.log*:

```text
hello world! job_1
```

*1_job_2_{timestamp}_stdout.log*:

```text
hello world! job_2
```


### Work with Airflow

We have added an event-based scheduler named `event_scheduler` to Airflow, so Flink AI Flow can also work with Airflow's event_scheduler, which is more powerful and has a Web UI to monitor the execution.

#### Prerequisites

1. MySQL

To start an Airflow server, you need to install and start a MySQL server in your machine. You need to create a database with specific character set in case of error when creating Apache Airflow tables, for example: 
```text
CREATE DATABASE airflow CHARACTER SET UTF8mb3 COLLATE utf8_general_ci;
```
Currently the AI Flow bundles a modified Airflow so users do not need to install the Apache Airflow manually.

#### Start notification server, Airflow Server and AI Flow Server
Run following command to start Notification service, AI Flow Server and Airflow Server:

```shell
start-aiflow.sh
```

If you execute this command for the first time, you will get the following output:

```text
The ${AIRFLOW_HOME}/airflow.cfg is not exists. You need to provide a mysql database to initialize the airflow, e.g.:
start-aiflow.sh mysql://root:root@127.0.0.1/airflow
```

Please prepare the MySQL database and rerun the `start-aiflow.sh` with the MySQL parameter.
If the servers start successfully, you will get the output like:

```text
Scheduler log: ${AIRFLOW_HOME}/scheduler.log
Scheduler pid: 69945
Web Server log: ${AIRFLOW_HOME}/web.log
Web Server pid: 69946
Master Server log:  ${AIRFLOW_HOME}/master_server.log
Master Server pid: 69947
Airflow deploy path: ${AIRFLOW_HOME}/airflow_deploy
Visit http://127.0.0.1:8080/ to access the airflow web server.
```

#### Prepare AI Flow Project

In order to properly adapt to the Airflow, the AI Flow project should have such a directory structure:

```text
SimpleProject
├─ project.yaml
├─ jar_dependencies
├─ resources
└─ python_codes
   ├─ __init__.py
   ├─ my_ai_flow.py
   └─ requirements.txt
```

For python jobs we only need to prepare the `python_codes` directory, the `resources` directory and the `project.yaml`.

Run following script to prepare a simple AI Flow project:

```shell
# this path should be determined according to your machine
export AIRFLOW_HOME=~/airflow

CURRENT_DIR=$(pwd)
AIRFLOW_DEPLOY_PATH="${AIRFLOW_HOME}/airflow_deploy"

# create the dir if not exists
mkdir SimpleProject >/dev/null 2>&1 || true
cd SimpleProject

# prepare the project.yaml
cat>project.yaml<<EOF
project_name: simple_project
master_ip: localhost
master_port: 50051
# for airflow scheduler this option should be enabled.
notification_uri: localhost:50052
airflow_deploy_path: ${AIRFLOW_DEPLOY_PATH}
EOF

# prepare the workflow_config.yaml
# create the dir if not exists
mkdir resources >/dev/null 2>&1 || true
cd resources
cat>workflow_config.yaml<<EOF
job_1:
  platform: local
  engine: python
  job_name: job_1

job_2:
  platform: local
  engine: python
  job_name: job_2

job_3:
  platform: local
  engine: python
  job_name: job_3
EOF
cd ../

# prepare the workflow python code
# create the dir if not exists
mkdir python_codes >/dev/null 2>&1 || true
cd python_codes
cat>airflow_dag_example.py<<EOF
import os
from typing import List

import ai_flow as af
from ai_flow import FunctionContext
from ai_flow.common.scheduler_type import SchedulerType
from python_ai_flow import Executor


class PrintHelloExecutor(Executor):
    def __init__(self, job_name):
        super().__init__()
        self.job_name = job_name

    def execute(self, function_context: FunctionContext, input_list: List) -> None:
        print("hello world! {}".format(self.job_name))


project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_workflow():
    with af.global_config_file(project_path + '/resources/workflow_config.yaml'):
        with af.config('job_1'):
            op_1 = af.user_define_operation(af.PythonObjectExecutor(PrintHelloExecutor('job_1')))

        with af.config('job_2'):
            op_2 = af.user_define_operation(af.PythonObjectExecutor(PrintHelloExecutor('job_2')))

        with af.config('job_3'):
            op_3 = af.user_define_operation(af.PythonObjectExecutor(PrintHelloExecutor('job_3')))

    af.stop_before_control_dependency(op_3, op_1)
    af.stop_before_control_dependency(op_3, op_2)


def run_workflow():
    build_workflow()
    af.set_project_config_file(project_path + '/project.yaml')
    # the airflow scheduler do not support waiting until the execution finished
    # so we just submit the workflow and exit
    af.run(project_path, dag_id='airflow_dag_example', scheduler_type=SchedulerType.AIRFLOW)


if __name__ == '__main__':
    run_workflow()

EOF
cd $CURRENT_DIR
```

#### Run the Workflow and Check the Execution Result

To run the workflow, just execute:

```shell
# this path should be determined according to your machine
export AIRFLOW_HOME=~/airflow

python3 SimpleProject/python_codes/airflow_dag_example.py
```

You can find the scheduled workflow on the [Airflow Web Server](http://127.0.0.1:8080/).

The outputs of each job can be found under `${AIRFLOW_HOME}/logs/airflow_dag_example/job_1/`, `${AIRFLOW_HOME}/logs/airflow_dag_example/job_2/` and `${AIRFLOW_HOME}/logs/airflow_dag_example/job_3/`.

#### Stop the Airflow Server and the AI Flow Master Server

Run following command to stop the servers:

```shell
stop-aiflow.sh
```
## Get Started in Docker
The Dockerfile is also provided, which helps users start a Flink AI Flow server with out-of-the-box environment. You can build an image like this:
```shell
docker build --rm -t flink-ai-extended/flink-ai-flow:v1 .
```

Before starting the container with the image, you need to make sure your MySQL server on your host machine has a valid database.
You can create the database using the following command in your MySQL CLI:
```SQL
CREATE DATABASE airflow CHARACTER SET UTF8mb3 COLLATE utf8_general_ci;
```
Also, if you are using MySQL 8.0+,  please also type following command in your MySQL CLI as well:
```SQL
ALTER USER 'username' IDENTIFIED WITH mysql_native_password BY 'password';
```

Then, to run the image, you need to pass your MySQL connection string as parameter, e.g.
```shell
 docker run -it -p 8080:8080 flink-ai-extended/flink-ai-flow:v1 mysql://user:password@127.0.0.1/airflow
```
Note, `127.0.0.1` should be replaced with `host.docker.internal` or any valid IP address which can be utilized by docker to access host machine's MySQL service.

To submit a workflow, you can run the following command:
```shell
python ${FLINK_AI_FLOW_SOURCES}/examples/quickstart_example/python_codes/airflow_dag_example.py
```
You can find the scheduled workflow on the [Airflow Web Server](http://127.0.0.1:8080/).
Once the workflow is done, you can check its correctness by viewing the output logs under `${AIRFLOW_HOME}/logs/airflow_dag_example` directory or via [Web UI](http://127.0.0.1:8080/). 
The Web UI should look like:

![](doc/images/docker_example2.png)

The Graph view is as follows:

![](doc/images/docker_example1.png)

If you meet any problems, please refer to the [Troubleshooting](#troubleshooting) section for help.

## Troubleshooting

#### 1. Fail on mysqlclient installation
According to mysqlclient's [document](https://github.com/PyMySQL/mysqlclient#install), extra steps are needed for installing mysqlclient with pip. Please check the document and take corresponding actions.

#### 2. `(2002, "Can't connect to MySQL server on '127.0.0.1' (115)")` when running in docker
Replace `mysql://user:password@127.0.0.1/airflow` with `mysql://user:password@host.docker.internal/airflow`.

#### 3. `Plugin caching_sha2_password could not be loaded:...` when running in docker
Due to MySQL's [document](https://dev.mysql.com/doc/refman/8.0/en/upgrading-from-previous-series.html), caching_sha2_password is the the default authentication plugin since MySQL 8.0. If you meet this problem 
when launching docker, you can fix it by changing it back to naive version. To do that, in your MySQL server on host machine, type following command:

```SQL
ALTER USER 'username' IDENTIFIED WITH mysql_native_password BY 'password';
```
Then restart MySQL service and docker image.


