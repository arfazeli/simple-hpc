#include "simple/config.hpp"

#include <fstream>
#include <stdexcept>

namespace {
std::string trim(const std::string& s) {
    const char* ws = " \t\r\n";
    auto b = s.find_first_not_of(ws);
    if (b == std::string::npos)
        return "";
    return s.substr(b, s.find_last_not_of(ws) - b + 1);
}
} // namespace

ConfigMap parse_config(std::istream& in) {
    ConfigMap m;
    std::string line;
    int n = 0;
    while (std::getline(in, line)) {
        ++n;
        auto c = line.find('#');
        if (c != std::string::npos)
            line.erase(c);
        line = trim(line);
        if (line.empty())
            continue;
        auto eq = line.find('=');
        std::string where = "config line " + std::to_string(n) + ": ";
        if (eq == std::string::npos)
            throw std::runtime_error(where + "malformed line (expected key = value)");
        std::string k = trim(line.substr(0, eq)), v = trim(line.substr(eq + 1));
        if (k.empty() || v.empty())
            throw std::runtime_error(where + "malformed line (empty key or value)");
        if (!m.emplace(k, v).second)
            throw std::runtime_error(where + "duplicate key '" + k + "'");
    }
    return m;
}

ConfigMap load_config(const std::string& path) {
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("cannot open config file: " + path);
    return parse_config(f);
}
