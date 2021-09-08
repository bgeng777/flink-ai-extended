# Demo Project
This document aims at showing how to run workflows in demo folder. Before running any example, please follow 
instructions in [Quick Start](https://github.com/alibaba/flink-ai-extended/wiki/Quick-Start) to make sure you 
have installed and started Flink AI Flow correctly.

## simple_workflow
This workflow is a `hello world` example. It contains 2 bash jobs, both of which will print a `hello` in the log of
[Web UI](localhost:8080)

Run the following commands in your terminal:
```shell
cd flink-ai-extended/flink-ai-flow/examples/demo
python workflows/simple_workflow/simple_workflow.py
```
If you are using IDEs(e.g. PyCharm), you can also click the `Run` button in the `simple_workflow.py` in your IDE.

## action_on_events
This workflow shows how to define a workflow based on the job status.

Run the following commands in your terminal:
```shell
cd flink-ai-extended/flink-ai-flow/examples/demo
python workflows/action_on_status/action_on_status.py.py
```
If you are using IDEs(e.g. PyCharm), you can also click the `Run` button in the `action_on_status.py` in your IDE.

## flink_sql_workflow
This workflow shows how to use the FlinkPythonProcessor and FlinkSqlProcessor to do model training and inference to 
define Flink jobs. 

You can run this workflow using mini cluster or standalone cluster by setting `run_mode` in the `flink_sql_workflow.yaml`.

If it is `local`, the prediction job will be run in mini cluster.

Run the following commands in your terminal:
```shell
cd flink-ai-extended/flink-ai-flow/examples/demo
python workflows/flink_sql_workflow/flink_sql_workflow.py
```

If it is `cluster`, the prediction job will be run in standalone cluster. The Java UDF in PyFlink is currently only supported 
in `cluster` mode. To use the Java UDF, users must compile and get the jar file.
Run the following commands in your terminal:
```shell
cd flink-ai-extended/flink-ai-flow/examples/demo
cd scalar-function 
# Compile the jar and put it under the project's dependencies/jar directory.
mvn clean package && cp target/scalar-function-1.0.jar ../dependencies/jar/
```

Then modify the `flink_sql_workflow.yaml`:
```yaml
predict:
  job_type: flink
  properties:
    run_mode: cluster
    flink_run_args: # The flink run command args(-pym etc.). It's type is List.
      - --jarfile 
      - scalar-function-1.0.jar # This must be the jar file name under dependencies/jar directory
```

Finally, run the following commands in your terminal:
```shell
cd flink-ai-extended/flink-ai-flow/examples/demo
python workflows/flink_sql_workflow/flink_sql_workflow.py
```

