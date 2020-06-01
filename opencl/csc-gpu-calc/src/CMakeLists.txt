file(GLOB_RECURSE MAIN
    "main.h"
    "main.cpp"
 )

source_group("Main" FILES ${MAIN})

set(SRC ${MAIN} PARENT_SCOPE)
