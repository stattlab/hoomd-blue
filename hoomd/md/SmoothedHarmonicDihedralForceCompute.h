// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/BondedGroupData.h"
#include "hoomd/ForceCompute.h"

#include <memory>

#include <vector>

/*! \file HarmonicDihedralForceCompute.h
    \brief Declares a class for computing harmonic dihedrals
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __SMOOTHEDHARMONICDIHEDRALFORCECOMPUTE_H__
#define __SMOOTHEDHARMONICDIHEDRALFORCECOMPUTE_H__

namespace hoomd
    {
namespace md
    {
struct dihedral_smoothed_harmonic_params
    {
    Scalar k;
    Scalar d;
    int n;
    Scalar phi_0;
    Scalar cos_on;
    Scalar cos_cut;
    Scalar Vs;
    int m;

#ifndef __HIPCC__
    dihedral_smoothed_harmonic_params() : k(0.), d(0.), n(0), phi_0(0.), cos_on(0.), cos_cut(0.), Vs(0.), m(0) { }

    dihedral_smoothed_harmonic_params(pybind11::dict v)
        : k(v["k"].cast<Scalar>()), d(v["d"].cast<Scalar>()), n(v["n"].cast<int>()),
          phi_0(v["phi0"].cast<Scalar>()),cos_on(v["smoothing_on"].cast<Scalar>()),
          cos_cut(v["smoothing_cut"].cast<Scalar>()),Vs(v["smoothing_Vs"].cast<Scalar>()),m(v["smoothing_m"].cast<int>())
        {
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["k"] = k;
        v["d"] = d;
        v["n"] = n;
        v["phi0"] = phi_0;
        v["smoothing_on"] = cos_on;
        v["smoothing_cut"] = cos_cut;
        v["smoothing_Vs"] = Vs;
        v["smoothing_m"] = m;
        return v;
        }
#endif
    } __attribute__((aligned(32)));

//! Computes harmonic dihedral forces on each particle
/*! Harmonic dihedral forces are computed on every particle in the simulation.

    The dihedrals which forces are computed on are accessed from ParticleData::getDihedralData
    \ingroup computes
*/
class PYBIND11_EXPORT SmoothedHarmonicDihedralForceCompute : public ForceCompute
    {
    public:
    //! Constructs the compute
    SmoothedHarmonicDihedralForceCompute(std::shared_ptr<SystemDefinition> sysdef);

    //! Destructor
    virtual ~SmoothedHarmonicDihedralForceCompute();

    //! Set the parameters
    virtual void
    setParams(unsigned int type, Scalar K, Scalar sign, int multiplicity, Scalar phi_0, Scalar cos_on, Scalar cos_cut, Scalar Vs, int m);

    virtual void setParamsPython(std::string type, pybind11::dict params);

    /// Get the parameters for a particular type
    pybind11::dict getParams(std::string type);

#ifdef ENABLE_MPI
    //! Get ghost particle fields requested by this pair potential
    /*! \param timestep Current time step
     */
    virtual CommFlags getRequestedCommFlags(uint64_t timestep)
        {
        CommFlags flags = CommFlags(0);
        flags[comm_flag::tag] = 1;
        flags |= ForceCompute::getRequestedCommFlags(timestep);
        return flags;
        }
#endif

    protected:
    Scalar* m_K;     //!< K parameter for multiple dihedral tyes
    Scalar* m_sign;  //!< sign parameter for multiple dihedral types
    int* m_multi;    //!< multiplicity parameter for multiple dihedral types
    Scalar* m_phi_0; //!< phi_0 parameter for multiple dihedral types

    Scalar* m_cos_on;
    Scalar* m_cos_cut;
    Scalar* m_VS;
    int* m_m;

    std::shared_ptr<DihedralData> m_dihedral_data; //!< Dihedral data to use in computing dihedrals


    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

    } // end namespace md
    } // end namespace hoomd

#endif
