#pragma once
#include "simple/field2d.hpp"
#include "simple/params.hpp"
#include "simple/solver.hpp"

#include <string>

void write_csv(const std::string& path, const Field2D& f);
// SPEC 3.8. ux, uy are resized to (NX, NY).
void cell_centre(const Params& p, const Field2D& u, const Field2D& v, Field2D& ux, Field2D& uy);
// u, v, p are full-grid fields (gathered on rank 0 under MPI).
void write_outputs(const Params& p, const Field2D& u, const Field2D& v, const Field2D& pf,
                   const RunResult& r, const std::string& cfg_path);
