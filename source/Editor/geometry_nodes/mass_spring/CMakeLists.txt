# c/cpp only
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} THISSRCS)

option(MASS_SPRING_ENABLE_OPENMP "Enable OpenMP acceleration for mass spring simulation" ON)
set(MASS_SPRING_OPENMP_TARGET "")
if(MASS_SPRING_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        set(MASS_SPRING_OPENMP_TARGET OpenMP::OpenMP_CXX)
        message(STATUS "[MassSpring] OpenMP enabled")
    else()
        message(STATUS "[MassSpring] OpenMP requested but not found; using serial code")
    endif()
endif()

# interface
add_library(mass_spring_utils INTERFACE)
target_link_libraries(mass_spring_utils INTERFACE
    stage nodes_system usd geometry usdShade Eigen3::Eigen autodiff igl::core igl_restricted::triangle ${MASS_SPRING_OPENMP_TARGET}
)
target_include_directories(mass_spring_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# static lib
if(THISSRCS)
    message(STATUS "[Found SRC for utility lib]")
    add_library(
        mass_spring_utils_static STATIC
        ${THISSRCS}
    )
    target_include_directories(mass_spring_utils_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(mass_spring_utils_static PUBLIC
        stage nodes_system usd geometry usdShade Eigen3::Eigen autodiff igl::core igl_restricted::triangle ${MASS_SPRING_OPENMP_TARGET}
    )

    # link to the interface
    target_link_libraries(
        mass_spring_utils INTERFACE
        mass_spring_utils_static
    )
endif()

set(mass_spring_utils_AVAILABLE TRUE PARENT_SCOPE)
