#pragma once
#include <istream>
#include <map>
#include <string>

using ConfigMap = std::map<std::string, std::string>;
ConfigMap parse_config(std::istream& in);
ConfigMap load_config(const std::string& path);
