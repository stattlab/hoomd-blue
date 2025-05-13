// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TwoStepConstantVolumeSLLODGPU.cuh"
#include "hip/hip_runtime.h"
#include <assert.h>

/*! \file TwoStepNVTGPU.cu
    \brief Defines GPU kernel code for NVT integration on the GPU. Used by TwoStepNVTGPU.
*/

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Takes the first 1/2 step forward in the NVT integration step
/*! \param d_pos array of particle positions
    \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_image array of particle images
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param work_size Number of members in the group for this GPU
    \param box Box dimensions for periodic boundary condition handling
    \param rescale_factor Velocity rescaling factor from thermostat
    \param deltaT Amount of real time to step forward in one time step

    Take the first half step forward in the NVT integration.

    See gpu_nve_step_one_kernel() for some performance notes on how to handle the group data reads
   efficiently.
*/
__global__ void gpu_nvt_sllod_rescale_step_one_kernel(Scalar4* d_pos,
                                                      Scalar4* d_vel,
                                                      const Scalar3* d_accel,
                                                      int3* d_image,
                                                      unsigned int* d_group_members,
                                                      unsigned int work_size,
                                                      BoxDim box,
                                                      Scalar rescale_factor,
                                                      Scalar deltaT,
                                                      Scalar shear_rate,
                                                      bool vel_correction,
                                                      bool flipped,
                                                      Scalar boundary_shear_velocity,
                                                      bool limit = false,
                                                      Scalar maximum_displacement = Scalar(0.))
    {
    // determine which particle this thread works on
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < work_size)
        {
        unsigned int idx = d_group_members[group_idx];

        // update positions to the next timestep and update velocities to the next half step
        Scalar4 postype = d_pos[idx];
        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);

        Scalar4 velmass = d_vel[idx];
        Scalar3 vel = make_scalar3(velmass.x, velmass.y, velmass.z);
        Scalar3 accel = d_accel[idx];

        // remove flow field
        vel.x -= shear_rate * pos.y;

        // rescale velocity
        vel *= rescale_factor;

        if (vel_correction == true)
            {
            // apply sllod velocity correction
            vel.x -= Scalar(0.5) * shear_rate * vel.y * deltaT;
            }

        // add flow field
        vel.x += shear_rate * pos.y;

        // update velocity
        vel += Scalar(0.5) * accel * deltaT;

        if (limit)
            {
            Scalar displacement = sqrtf(dot(vel, vel));
            if (displacement * deltaT > maximum_displacement)
                vel = vel * maximum_displacement / displacement * deltaT;
            }

        pos += vel * deltaT;

        // read in the image flags
        int3 image = d_image[idx];

        // if box deformation caused a flip, wrap pos back into box
        if (flipped)
            {
            image.x += image.y;
            }

        // time to fix the periodic boundary conditions
        box.wrap(pos, image);

        // Periodic boundary correction to velocity:
        // if particle leaves from (+/-) y boundary it gets (-/+) velocity at boundary
        // NOTE: pair potentials dependent on differences in
        // velocities (e.g. DPD) are not supported.

        if ((image.y - d_image[idx].y) == 1) // crossed pbc in +y, image increased by 1
            {
            vel.x -= boundary_shear_velocity;
            }
        else if ((image.y - d_image[idx].y) == -1) // crossed pbc in -y, image decreased by 1
            {
            vel.x += boundary_shear_velocity;
            }

        // write out the results
        d_pos[idx] = make_scalar4(pos.x, pos.y, pos.z, postype.w);
        d_vel[idx] = make_scalar4(vel.x, vel.y, vel.z, velmass.w);
        d_image[idx] = image;
        }
    }

/*! \param d_pos array of particle positions
    \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_image array of particle images
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param box Box dimensions for periodic boundary condition handling
    \param block_size Size of the block to run
    \param rescale_factor Thermostat rescaling factor
    \param deltaT Amount of real time to step forward in one time step
*/
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
                                          bool use_limit,
                                          Scalar maximum_displacement)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_nvt_sllod_rescale_step_one_kernel);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    unsigned int nwork = group_size;

    // setup the grid to run the kernel
    dim3 grid((nwork / run_block_size) + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    // run the kernel
    hipLaunchKernelGGL((gpu_nvt_sllod_rescale_step_one_kernel),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       d_pos,
                       d_vel,
                       d_accel,
                       d_image,
                       d_group_members,
                       nwork,
                       box,
                       rescale_factor,
                       deltaT,
                       shear_rate,
                       vel_correction,
                       flipped,
                       boundary_shear_velocity,
                       use_limit,
                       maximum_displacement);

    return hipSuccess;
    }

//! Takes the second 1/2 step forward in the NVT integration step
/*! \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param work_size Number of members in the group for this GPU
    \param d_net_force Net force on each particle
    \param deltaT Amount of real time to step forward in one time step
*/
__global__ void gpu_nvt_sllod_rescale_step_two_kernel(Scalar4* d_vel,
                                                      Scalar3* d_accel,
                                                      unsigned int* d_group_members,
                                                      unsigned int work_size,
                                                      Scalar4* d_net_force,
                                                      Scalar deltaT,
                                                      Scalar rescale_factor,
                                                      Scalar shear_rate,
                                                      bool vel_correction)
    {
    // determine which particle this thread works on
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < work_size)
        {
        unsigned int idx = d_group_members[group_idx];

        // read in the net force and calculate the acceleration
        Scalar4 net_force = d_net_force[idx];
        Scalar3 accel = make_scalar3(net_force.x, net_force.y, net_force.z);

        Scalar4 vel = d_vel[idx];
        Scalar3 v = make_scalar3(vel.x, vel.y, vel.z);

        Scalar mass = vel.w;
        accel = accel / mass;

        Scalar3 v_del_u = make_scalar3(0.0, 0.0, 0.0);

        if (vel_correction == true)
            {
            // SLLOD correction to velocity: shear rate tensor dotted with velocity
            v_del_u = make_scalar3(shear_rate * vel.y, 0.0, 0.0);
            }

        // update velocity
        v += Scalar(0.5) * (accel - v_del_u) * deltaT;

        // rescale
        v *= rescale_factor;

        // save
        d_vel[idx] = make_scalar4(v.x, v.y, v.z, vel.w);

        // since we calculate the acceleration, we need to write it for the next step
        d_accel[idx] = accel;
        }
    }

/*! \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Net force on each particle
    \param block_size Size of the block to execute on the device
    \param deltaT Amount of real time to step forward in one time step
    \param rescale_factor Exponential velocity scaling factor
*/
hipError_t gpu_nvt_sllod_rescale_step_two(Scalar4* d_vel,

                                          Scalar3* d_accel,
                                          unsigned int* d_group_members,
                                          unsigned int group_size,
                                          Scalar4* d_net_force,
                                          unsigned int block_size,
                                          Scalar deltaT,
                                          Scalar rescale_factor,
                                          Scalar shear_rate)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_nvt_sllod_rescale_step_two_kernel);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    unsigned int nwork = group_size;

    // setup the grid to run the kernel
    dim3 grid((nwork / run_block_size) + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    // run the kernel
    hipLaunchKernelGGL((gpu_nvt_sllod_rescale_step_two_kernel),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       d_vel,
                       d_accel,
                       d_group_members,
                       nwork,
                       d_net_force,
                       deltaT,
                       rescale_factor,
                       shear_rate);

    return hipSuccess;
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
