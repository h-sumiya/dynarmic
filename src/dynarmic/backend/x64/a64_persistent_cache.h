/* SPDX-License-Identifier: 0BSD */
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace Dynarmic::A64::Persistent {

// Only machine code, relocation records and instruction-validation metadata
// are stored. Restoring a region does not run the translator or optimizer.
struct Reference {
    std::uint32_t image = 0;
    std::uint64_t offset = 0;
    std::uint64_t flags = 0;
};
struct HostImage {
    std::string name;
    std::string identity;
    std::uint64_t base = 0, size = 0;
};
struct Relocation {
    std::uint64_t offset = 0;
    std::uint8_t kind = 0;
    Reference target;
};
struct Word { Reference address; std::uint32_t instruction = 0; };
struct Block {
    Reference location;
    std::uint64_t entry = 0, size = 0;
    std::vector<Word> words;
};
struct Link { Reference location; std::uint64_t cell = 0, fallback = 0; };
struct Fault { std::uint64_t rip = 0, resume = 0, callback = 0; };
struct BlockIndex {
    Reference location;
    std::uint64_t entry = 0, size = 0;
    std::uint64_t record_offset = 0, record_size = 0, record_checksum = 0;
};
struct File;
struct Index {
    std::string runtime_identity;
    std::uint64_t contract = 0, features = 0, capacity = 0, prelude = 0, data_offset = 0, body_end = 0;
    std::uint64_t data_file_offset = 0, data_size = 0, data_checksum = 0;
    std::uint64_t read_bytes = 0, file_size = 0;
    std::vector<std::string> images;
    std::vector<HostImage> hosts;
    std::vector<BlockIndex> blocks;
    std::shared_ptr<File> file;
};
struct BlockRecord {
    Block block;
    std::vector<std::uint8_t> body;
    std::vector<Relocation> relocations;
    std::vector<Link> links;
    std::vector<Fault> faults;
    std::uint64_t read_bytes = 0;
};
struct Region {
    std::string runtime_identity;
    std::uint64_t contract = 0, features = 0, capacity = 0, prelude = 0, data_offset = 0;
    std::vector<std::string> images;
    std::vector<HostImage> hosts;
    std::vector<std::uint8_t> data, body;
    std::vector<Relocation> relocations;
    std::vector<Block> blocks;
    std::vector<Link> links;
    std::vector<Fault> faults;
};

HostImage HostImageAt(std::uint64_t address);
HostImage ResolveHostImage(const HostImage& image);
std::string RuntimeIdentity();
std::uint64_t Hash(std::span<const std::uint8_t> bytes, std::uint64_t seed = 14695981039346656037ULL);
std::string Key(const Region& region);
Index ReadIndex(const std::filesystem::path& path, std::uint64_t maximum_bytes);
std::vector<std::uint8_t> ReadData(const Index& index);
BlockRecord ReadBlock(const Index& index, std::size_t ordinal);
void Write(const std::filesystem::path& path, const Region& region);
void Trim(const std::filesystem::path& directory, std::uint64_t maximum_bytes);

} // namespace Dynarmic::A64::Persistent
