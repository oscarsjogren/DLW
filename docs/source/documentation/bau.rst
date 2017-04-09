===
BAU
===

The :class:`ez_climate.bau.DLWBusinessAsUsual` provides an analysis framework of business as usual scenario. We assume constant consumption growth and GHG emissions that grow linearly over time without mitigation. For analysis, emission level are given at certain decision time points. Emissions between those decision time points are calcualted using linear interploation. GHG levels are calculated in accordance which the emission path.

If a different assumption about business as usual scenario (e.g., non-linear growth of GHG emission) is made, you can created an inheritance from the base class and define new interplotation method in function :func:`emission_by_time`.



