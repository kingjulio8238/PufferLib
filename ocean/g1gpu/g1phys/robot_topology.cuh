// Robot selector — the per-robot compile point. Defaults to G1; a Go2 build
// passes -DROBOT_TOPO_H='"go2_topology.cuh"'. Symbol names stay G1_*/g1c_*
// (legacy = "the compiled robot").
#pragma once
#ifndef ROBOT_TOPO_H
#define ROBOT_TOPO_H "g1_topology.cuh"
#endif
#include ROBOT_TOPO_H
