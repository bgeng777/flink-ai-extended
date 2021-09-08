package com.pyflink.table;

import org.apache.flink.table.functions.ScalarFunction;

public class DummyJavaUDF extends ScalarFunction {

    public float eval(float a) {
        return a * 1;
    }
}
