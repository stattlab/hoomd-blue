// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// inclusion guard
#pragma once

#include <cmath>
#include <hoomd/RandomNumbers.h>
#include <hoomd/Updater.h>
#include <hoomd/Variant.h>
#include <hoomd/VectorVariant.h>

#include "IntegratorHPMC.h"

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace hpmc
    {
/** Cylinder Twist algorithm
*/
class UpdaterCylinderRotate : public Updater
    {
    public:
    /** Constructor

        @param sysdef System definition
        @param mc HPMC integrator object
        @param max_overlaps_per_particle Maximum number of overlaps allowed per particle
        @param g The minimum scale factor to use when scaling the box parameters
    */
    UpdaterCylinderRotate(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<Trigger> trigger,
                         std::shared_ptr<IntegratorHPMC> mc,
                         double max_overlaps_per_particle,
                         double g
                         );

    /// Destructor
    virtual ~UpdaterCylinderRotate();

    /** Take one timestep forward

        @param timestep timestep at which update is being evaluated
    */
    virtual void update(uint64_t timestep);

    /// Get the maximum number of overlaps allowed per particle
    double getMaxOverlapsPerParticle()
        {
        return m_max_overlaps_per_particle;
        }

    /// Set the maximum number of overlaps allowed per particle
    void setMaxOverlapsPerParticle(double max_overlaps_per_particle)
        {
        m_max_overlaps_per_particle = max_overlaps_per_particle;
        }

    /// Get the g scale factor
    double getG()
        {
        return m_g;
        }

    /// Set g 
    void setG(double g)
        {
        m_g = g;
        }

    /// Return true if the updater is complete and the simulation should end.
    virtual bool isComplete()
        {
        return m_is_complete;
        }

    /// Set the RNG instance
    void setInstance(unsigned int instance)
        {
        m_instance = instance;
        }

    /// Get the RNG instance
    unsigned int getInstance()
        {
        return m_instance;
        }

    private:
    /// HPMC integrator object
    std::shared_ptr<IntegratorHPMC> m_mc;

    /// Maximum number of overlaps allowed per particle
    double m_max_overlaps_per_particle;

    /// Minimum scale factor to use when scaling box parameters
    double m_g;

    /// Unique ID for RNG seeding
    unsigned int m_instance = 0;

    /// Perform the alpha move
    void performAlphaMove(uint64_t timestep, quat<Scalar>& m_aq, Scalar g);

    /// Store the last HPMC counters
    hpmc_counters_t m_last_move_counters;

    /// Track whether the compression is complete
    bool m_is_complete = false;

    };

namespace detail
    {
/// Export UpdaterCylinderRotate to Python
void export_UpdaterCylinderRotate(pybind11::module& m);
    } // end namespace detail
    } // end namespace hpmc
    } // end namespace hoomd
