// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TwoStepConstantVolumeSLLODGPU.h"
#include "TwoStepConstantVolumeSLLODGPU.cuh"
#include "TwoStepNVEGPU.cuh"
#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#include "hoomd/HOOMDMPI.h"
#endif

namespace hoomd::md
    {

TwoStepConstantVolumeSLLODGPU::TwoStepConstantVolumeSLLODGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                   std::shared_ptr<ParticleGroup> group,
                                                   std::shared_ptr<Thermostat> thermostat,
                                                   Scalar shear_rate)
    : TwoStepConstantVolumeSLLOD(sysdef, group, thermostat, shear_rate)
    {
        std::cout<< "in TwoStepConstantVolumeSLLODGPU create GPU"<< std::endl;
    }

void TwoStepConstantVolumeSLLODGPU::integrateStepOne(uint64_t timestep)
    {
    std::cout<< "in TwoStepConstantVolumeSLLODGPU::integrateStepOne GPU"<< std::endl;
    if (m_group->getNumMembersGlobal() == 0)
        {
        throw std::runtime_error("Empty integration group.");
        }

    unsigned int group_size = m_group->getNumMembers();
    const auto&& rescalingFactors = m_thermostat
                                        ? m_thermostat->getRescalingFactorsOne(timestep, m_deltaT)
                                        : std::array<Scalar, 2> {1., 1.};
        {
        // access all the needed data
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(),
                                   access_location::device,
                                   access_mode::readwrite);
        ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(),
                                   access_location::device,
                                   access_mode::readwrite);
        ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(),
                                     access_location::device,
                                     access_mode::read);
        ArrayHandle<int3> d_image(m_pdata->getImages(),
                                  access_location::device,
                                  access_mode::readwrite);

        BoxDim box = m_pdata->getBox();
        ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(),
                                                access_location::device,
                                                access_mode::read);

        auto limits = getKernelLimitValues(timestep);

        // box deformation: update tilt factor of global box
        bool flipped = deformGlobalBox();
        std::cout<< "after deformGlobalBox "<< std::endl;
        m_exec_conf->setDevice();
        std::cout<< "after setDevice "<< std::endl;
        // perform the update on the GPU
        m_tuner_one->begin();

        std::cout<< "before  gpu_nvt_sllod_rescale_step_one"<< std::endl;
        kernel::gpu_nvt_sllod_rescale_step_one(d_pos.data,
                                         d_vel.data,
                                         d_accel.data,
                                         d_image.data,
                                         d_index_array.data,
                                         group_size,
                                         box,
                                         m_tuner_one->getParam()[0],
                                         rescalingFactors[0], // m_exp_thermo_fac,
                                         m_deltaT,
                                         m_shear_rate,
                                         flipped,
                                         m_boundary_shear_velocity,
                                         limits.first,
                                         limits.second);

        std::cout<< "after gpu_nvt_sllod_rescale_step_one "<< std::endl;
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_one->end();
        }

    if (m_aniso)
        {
        // angular degrees of freedom, step one
        ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(),
                                           access_location::device,
                                           access_mode::readwrite);
        ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(),
                                      access_location::device,
                                      access_mode::readwrite);
        ArrayHandle<Scalar4> d_net_torque(m_pdata->getNetTorqueArray(),
                                          access_location::device,
                                          access_mode::read);
        ArrayHandle<Scalar3> d_inertia(m_pdata->getMomentsOfInertiaArray(),
                                       access_location::device,
                                       access_mode::read);
        ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(),
                                                access_location::device,
                                                access_mode::read);

        m_tuner_angular_one->begin();
        kernel::gpu_nve_angular_step_one(d_orientation.data,
                                         d_angmom.data,
                                         d_inertia.data,
                                         d_net_torque.data,
                                         d_index_array.data,
                                         m_group->getNumMembers(),
                                         m_deltaT,
                                         rescalingFactors[1],
                                         m_tuner_angular_one->getParam()[0]);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_angular_one->end();
        }
    std::cout<< "before advanceThermostat "<< std::endl;
    // advance thermostat
    if (m_thermostat)
        {
        m_thermostat->advanceThermostat(timestep, m_deltaT, m_aniso);
        }
    }

void TwoStepConstantVolumeSLLODGPU::integrateStepTwo(uint64_t timestep)
    {
    std::cout<< "in TwoStepConstantVolumeSLLODGPU::integrateStepTwo GPU"<< std::endl;
    unsigned int group_size = m_group->getNumMembers();

    const GPUArray<Scalar4>& net_force = m_pdata->getNetForce();

    ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);
    const auto&& rescalingFactors = m_thermostat
                                        ? m_thermostat->getRescalingFactorsTwo(timestep, m_deltaT)
                                        : std::array<Scalar, 2> {1., 1.};

        {
        ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(),
                                   access_location::device,
                                   access_mode::readwrite);
        ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(),
                                     access_location::device,
                                     access_mode::readwrite);
        ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);

        m_exec_conf->setDevice();

        // perform the update on the GPU
        m_tuner_two->begin();
        kernel::gpu_nvt_sllod_rescale_step_two(d_vel.data,
                                         d_accel.data,
                                         d_index_array.data,
                                         group_size,
                                         d_net_force.data,
                                         m_tuner_two->getParam()[0],
                                         m_deltaT,
                                         rescalingFactors[0],
                                         m_shear_rate);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_two->end();
        }

    if (m_aniso)
        {
        // second part of angular update
        ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(),
                                           access_location::device,
                                           access_mode::read);
        ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(),
                                      access_location::device,
                                      access_mode::readwrite);
        ArrayHandle<Scalar4> d_net_torque(m_pdata->getNetTorqueArray(),
                                          access_location::device,
                                          access_mode::read);
        ArrayHandle<Scalar3> d_inertia(m_pdata->getMomentsOfInertiaArray(),
                                       access_location::device,
                                       access_mode::read);

        m_tuner_angular_two->begin();
        kernel::gpu_nve_angular_step_two(d_orientation.data,
                                         d_angmom.data,
                                         d_inertia.data,
                                         d_net_torque.data,
                                         d_index_array.data,
                                         m_group->getNumMembers(),
                                         m_deltaT,
                                         rescalingFactors[1],
                                         m_tuner_angular_two->getParam()[0]);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_angular_two->end();
        }
    }
    } // namespace hoomd::md

namespace hoomd::md::detail
    {
void export_TwoStepConstantVolumeSLLODGPU(pybind11::module& m)
    {
    pybind11::class_<TwoStepConstantVolumeSLLODGPU,
                     TwoStepConstantVolumeSLLOD,
                     std::shared_ptr<TwoStepConstantVolumeSLLODGPU>>(m, "TwoStepConstantVolumeSLLODGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            std::shared_ptr<Thermostat>,
                            Scalar>());
    }
    } // namespace hoomd::md::detail
