#pragma once
#include "simple/field2d.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Reads comma-separated rows of doubles. Throws on a missing file, a bad number or ragged rows.
inline std::vector<std::vector<double>> load_csv_rows(const std::string& path, bool skip_header) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open: " + path);
    std::vector<std::vector<double>> rows;
    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
        if (first && skip_header) {
            first = false;
            continue;
        }
        first = false;
        if (line.empty())
            continue;
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            char* end = nullptr;
            double x = std::strtod(cell.c_str(), &end);
            if (end == cell.c_str() || *end != '\0')
                throw std::runtime_error("bad number '" + cell + "' in " + path);
            row.push_back(x);
        }
        if (!rows.empty() && row.size() != rows.front().size())
            throw std::runtime_error("ragged rows in " + path);
        rows.push_back(std::move(row));
    }
    return rows;
}

inline Field2D load_csv(const std::string& path) {
    auto rows = load_csv_rows(path, false);
    if (rows.empty())
        throw std::runtime_error("empty csv: " + path);
    Field2D f(static_cast<int>(rows.size()), static_cast<int>(rows[0].size()));
    for (int i = 0; i < f.rows(); ++i)
        for (int j = 0; j < f.cols(); ++j)
            f(i, j) = rows[i][j];
    return f;
}
