# Standalone tests avoid downloads and keep assert() enabled in Release builds.
foreach(test experiment_options_test recall_evaluator_test)
    add_executable(${test} ${test}.cpp)
    target_link_libraries(${test} PRIVATE coraanns_host)
    target_compile_options(${test} PRIVATE -UNDEBUG)
    add_test(NAME ${test} COMMAND ${test})
    set_tests_properties(${test} PROPERTIES LABELS cpu ENVIRONMENT "OMP_NUM_THREADS=2")
endforeach()

find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(NAME experiment_runner COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_experiment_runner.py")
    set_tests_properties(experiment_runner PROPERTIES LABELS cpu TIMEOUT 60)
endif()
if(Python3_Interpreter_FOUND AND CORAANNS_BUILD_BASELINES)
    add_test(NAME cpu_pipeline COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/smoke_cpu.py"
        --indexer $<TARGET_FILE:coraanns_build_index>
        --search $<TARGET_FILE:rabitq_cpu>)
    set_tests_properties(cpu_pipeline PROPERTIES LABELS cpu TIMEOUT 120
        ENVIRONMENT "OMP_NUM_THREADS=2;OPENBLAS_NUM_THREADS=2")
endif()

if(CORAANNS_BUILD_GPU_TESTS)
    add_executable(gpu_cpu_highk_vanguard_test gpu_cpu_highk_vanguard_test.cpp)
    target_link_libraries(gpu_cpu_highk_vanguard_test PRIVATE coraanns_highk)
    add_test(NAME gpu_cpu_highk_vanguard_test COMMAND gpu_cpu_highk_vanguard_test)
    set_tests_properties(gpu_cpu_highk_vanguard_test PROPERTIES LABELS gpu)
    if(CORAANNS_BUILD_BASELINES)
        add_executable(gpu_index_residency_test gpu_index_residency_test.cpp)
        target_link_libraries(gpu_index_residency_test PRIVATE rabitq_gpu_kernels)
        add_executable(gpu_highk_validation gpu_highk_validation.cpp)
        target_link_libraries(gpu_highk_validation PRIVATE rabitq_gpu_highk_kernels)
        foreach(test gpu_index_residency_test gpu_highk_validation)
            add_test(NAME ${test} COMMAND ${test})
            set_tests_properties(${test} PROPERTIES LABELS gpu)
        endforeach()
    endif()
endif()
