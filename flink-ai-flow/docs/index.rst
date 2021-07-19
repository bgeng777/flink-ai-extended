.. Flink AI Flow documentation master file, created by
   sphinx-quickstart on Fri Jul 16 10:18:41 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Flink AI Flow's Documentation!
=========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents
   
   source/ai_flow
   source2/ai_flow_plugins

====================
Core Classes/Modules
====================

   :class:`ai_flow.ai_graph.ai_graph.AIGraph`

   Core abstraction of Flink AI Flow. Workflows defined by users will be translated into AIGraph by the Flink AI Flow framework.
   AIGraph consists of AINode and edges. For edges, they are either the :class:`~ai_flow.ai_graph.data_edge.DataEdge`  between AINodes in a job 
   or the :class:`~ai_flow.workflow.control_edge.ControlEdge` between jobs.

   :py:mod:`ai_flow.api.ops`

   Main module for defining customized workflow. It provides users with a variety of methods(e.g. :py:meth:`~ai_flow.api.ops.transform`, :py:meth:`~ai_flow.api.ops.train`) to define their own machine learning workflow.
   

   :py:mod:`ai_flow.api.workflow_operation`

   Module for manipulating workflows including managing a workflow's scheduling and execution.

