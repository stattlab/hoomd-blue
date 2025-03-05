// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ComputeThermoGPU.cuh"
#include "ComputeThermoSLLODGPU.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include <hip/hip_runtime.h>

#include <assert.h>

/*! \file ComputeThermoGPU.cu
    \brief Defines GPU kernel code for computing thermodynamic properties on the GPU. Used by
   ComputeThermoGPU.
*/

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
// TODO: having two kernels for add and remove is very readable and clear, but the
//  remove/add kernels differ in only  a single sign. Should they be joined into one
//  kernel with a bool or a sign to change  whether its adding or removing to avoid
//  code dublications?

// TODO: this kernel does not work and is not tested for multi GPU
__global__ void gpu_remove_flow_field_kernel(Scalar4* d_pos,
                                             Scalar4* d_vel,
                                             unsigned int* d_group_members,
                                             Scalar shear_rate,
                                             unsigned int group_size)
    {
    // determine which particle this thread works on
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];

        Scalar4 pos = d_pos[idx];
        Scalar4 vel = d_vel[idx];

        vel.x -= shear_rate * pos.y;

        // write out data
        d_vel[idx] = vel; // make_scalar4(v.x,v.y,v.z,vel.w);
        }
    }

// TODO: this kernel does not work and is not tested for multi GPU
__global__ void gpu_add_flow_field_kernel(Scalar4* d_pos,
                                          Scalar4* d_vel,
                                          unsigned int* d_group_members,
                                          Scalar shear_rate,
                                          unsigned int group_size)

    {
    // determine which particle this thread works on
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];

        Scalar4 pos = d_pos[idx];
        Scalar4 vel = d_vel[idx];

        vel.x += shear_rate * pos.y;

        // write out data
        d_vel[idx] = vel; // make_scalar4(v.x,v.y,v.z,vel.w);
        }
    }

hipError_t gpu_remove_flow_field(Scalar4* d_pos,
                                 Scalar4* d_vel,
                                 unsigned int* d_group_members,
                                 Scalar shear_rate,
                                 unsigned int group_size)
    {
    // setup the grid to run the kernel
    dim3 grid(group_size / block_size + 1, 1, 1);
    dim3 threads(block_size, 1, 1);
    // run the kernel
    hipLaunchKernelGGL(gpu_remove_flow_field_kernel,
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       d_pos,
                       d_vel,
                       d_group_members,
                       shear_rate,
                       group_size);

    return hipSuccess;
    }

hipError_t gpu_add_flow_field(Scalar4* d_pos,
                              Scalar4* d_vel,
                              unsigned int* d_group_members,
                              Scalar shear_rate,
                              unsigned int group_size)
    {
    // setup the grid to run the kernel
    dim3 grid(group_size / block_size + 1, 1, 1);
    dim3 threads(block_size, 1, 1);
    // run the kernel
    hipLaunchKernelGGL(gpu_add_flow_field_kernel,
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       d_pos,
                       d_vel,
                       d_group_members,
                       shear_rate,
                       group_size);

    return hipSuccess;
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
