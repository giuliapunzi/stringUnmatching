#include "matcher.hpp"
#include <sstream>
#include <utility>

using namespace strum;

Matcher::Matcher(const std::string& bytes, byte_t excess)
        : bytes_(bytes), length_(bytes_.size()), excess_(excess) {};

Matcher::Matcher(std::string&& bytes, byte_t excess)
        : bytes_(std::move(bytes)), length_(bytes_.size()), excess_(excess) {};
