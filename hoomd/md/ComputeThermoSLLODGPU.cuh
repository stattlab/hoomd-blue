// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef _COMPUTE_THERMO_SLLOD_GPU_CUH_
#define _COMPUTE_THERMO_SLLOD_GPU_CUH_

#include "ComputeThermoTypes.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"
#include "ComputeThermoGPU.cuh"

/*! \file ComputeThermoGPU.cuh
    \brief Kernel driver function declarations for ComputeThermoGPU
    */

namespace hoomd
    {
namespace md
    {
namespace kernel
    {

hipError_t gpu_remove_flow_field(Scalar4* d_pos,
                                 Scalar4* d_vel,
                                 unsigned int* d_group_members,
                                 Scalar shear_rate,
                                 unsigned int group_size);


hipError_t gpu_add_flow_field(Scalar4* d_pos,
                              Scalar4* d_vel,
                              unsigned int* d_group_members,
                              Scalar shear_rate,
                              unsigned int group_size);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif
