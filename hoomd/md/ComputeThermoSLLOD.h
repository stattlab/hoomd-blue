// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ComputeThermoTypes.h"
#include "ComputeThermo.h"
#include "hoomd/ParticleGroup.h"

#include <limits>
#include <memory>

/*! \file ComputeThermo.h
    \brief Declares a class for computing thermodynamic quantities
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __COMPUTE_THERMO_SLLOD_H__
#define __COMPUTE_THERMO_SLLOD_H__

namespace hoomd
    {
namespace md
    {
//! Computes thermodynamic properties of a group of particles with streaming velocity field.
/*! ComputeThermo calculates instantaneous thermodynamic properties for the case with steaming velocity applied.
    It first removes the streaming velocity, computes the thermodynamic properties, and then adds the
    streaming velocity back.

    \ingroup computes
*/
class PYBIND11_EXPORT ComputeThermoSLLOD : public ComputeThermo
    {
    public:
    //! Constructs the compute
    ComputeThermoSLLOD(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<ParticleGroup> group, Scalar shear_rate);

    //! Destructor
    virtual ~ComputeThermoSLLOD();

    //! Does the actual computation
    virtual void computeProperties();

    protected:
    Scalar m_shear_rate;
    };

    } // end namespace md
    } // end namespace hoomd

#endif
