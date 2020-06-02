file(GLOB_RECURSE LIBS
    "cxxopts.h"
    "logger.h"
)

source_group("Libs" FILES ${LIBS})

set(LIBS ${LIBS} PARENT_SCOPE)
