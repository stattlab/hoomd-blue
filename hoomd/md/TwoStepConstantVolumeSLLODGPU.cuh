// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

#ifndef HOOMD_TWOSTEPNVTBASEGPU_SLLOD_CUH
#define HOOMD_TWOSTEPNVTBASEGPU_SLLOD_CUH

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel driver for the first part of the NVT update called by TwoStepNVTGPU
hipError_t gpu_nvt_sllod_rescale_step_one(Scalar4* d_pos,
                                          Scalar4* d_vel,
                                          const Scalar3* d_accel,
                                          int3* d_image,
                                          unsigned int* d_group_members,
                                          unsigned int group_size,
                                          const BoxDim& box,
                                          unsigned int block_size,
                                          Scalar rescale_factor,
                                          Scalar deltaT,
                                          Scalar shear_rate,
                                          bool vel_correction,
                                          bool flipped,
                                          Scalar boundary_shear_velocity,
                                          bool limit = false,
                                          Scalar global_hi_y,
                                          Scalar global_lo_y,
                                          Scalar limit_displacement = Scalar(0.));

//! Kernel driver for the second part of the NVT update called by NVTUpdaterGPU
hipError_t gpu_nvt_sllod_rescale_step_two(Scalar4* d_vel,
                                          Scalar3* d_accel,
                                          unsigned int* d_group_members,
                                          unsigned int group_size,
                                          Scalar4* d_net_force,
                                          unsigned int block_size,
                                          Scalar deltaT,
                                          Scalar rescale_factor,
                                          Scalar shear_rate,
                                          bool vel_correction);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
#endif // HOOMD_TWOSTEPNVTBASEGPU_CUH
