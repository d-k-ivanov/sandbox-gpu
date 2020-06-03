file(GLOB_RECURSE MAIN
    "main.h"
    "main.cpp"
 )

source_group("Main" FILES ${MAIN})

file(GLOB_RECURSE TASKS
    "task1.cpp"
    # "task2.cpp"
    # "task3.cpp"
 )

source_group("Tasks" FILES ${TASKS})

set(SRC ${MAIN} ${TASKS} PARENT_SCOPE)
