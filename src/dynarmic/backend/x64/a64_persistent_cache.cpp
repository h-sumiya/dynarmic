/* SPDX-License-Identifier: 0BSD */
#include "dynarmic/backend/x64/a64_persistent_cache.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#include <link.h>
#include <unistd.h>
#endif

namespace Dynarmic::A64::Persistent {
namespace {
constexpr std::array<std::uint8_t, 8> magic{'A', '2', 'E', 'A', '6', '4', 'C', 1};
constexpr std::size_t maximum_items = 4 * 1024 * 1024;

std::string FileIdentity(const std::filesystem::path& path) {
    const auto metadata = std::to_string(std::filesystem::file_size(path)) + ":" +
        std::to_string(std::filesystem::last_write_time(path).time_since_epoch().count());
    struct CachedIdentity { std::string metadata, identity; };
    static std::mutex mutex;
    static std::unordered_map<std::string, CachedIdentity> cache;
    const auto key = std::filesystem::absolute(path).lexically_normal().string();
    std::lock_guard lock(mutex);
    const auto found = cache.find(key);
    if (found != cache.end() && found->second.metadata == metadata) return found->second.identity;
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot read A64 host image identity");
    std::array<std::uint8_t, 65536> bytes;
    std::uint64_t digest = 14695981039346656037ULL;
    while (input) {
        input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
        digest = Hash(std::span(bytes).first(static_cast<std::size_t>(input.gcount())), digest);
    }
    if (!input.eof()) throw std::runtime_error("Cannot fingerprint A64 host image");
    const auto identity = metadata + ":" + std::to_string(digest);
    cache[key] = {metadata, identity};
    return identity;
}

struct Writer {
    std::vector<std::uint8_t> bytes;
    template<class T> void Integer(T value) {
        static_assert(std::is_unsigned_v<T>);
        for (std::size_t i = 0; i < sizeof(T); ++i)
            bytes.push_back(static_cast<std::uint8_t>(value >> (i * 8)));
    }
    void Bytes(std::span<const std::uint8_t> data) {
        Integer<std::uint64_t>(data.size());
        bytes.insert(bytes.end(), data.begin(), data.end());
    }
    void String(const std::string& value) {
        Bytes({reinterpret_cast<const std::uint8_t*>(value.data()), value.size()});
    }
    void Ref(const Reference& ref) {
        Integer(ref.image); Integer(ref.offset); Integer(ref.flags);
    }
    template<class T, class F> void Vector(const std::vector<T>& values, F write) {
        Integer<std::uint64_t>(values.size());
        for (const auto& value : values) write(value);
    }
};

struct Reader {
    std::span<const std::uint8_t> bytes;
    template<class T> T Integer() {
        static_assert(std::is_unsigned_v<T>);
        if (bytes.size() < sizeof(T)) throw std::runtime_error("Truncated A64 machine-code cache");
        T result = 0;
        for (std::size_t i = 0; i < sizeof(T); ++i) result |= T(bytes[i]) << (8 * i);
        bytes = bytes.subspan(sizeof(T));
        return result;
    }
    std::vector<std::uint8_t> Bytes() {
        const auto size = Integer<std::uint64_t>();
        if (size > bytes.size()) throw std::runtime_error("Invalid A64 cache byte range");
        std::vector<std::uint8_t> result(bytes.begin(), bytes.begin() + size);
        bytes = bytes.subspan(size);
        return result;
    }
    std::string String() {
        auto value = Bytes();
        if (value.size() > 65536) throw std::runtime_error("Invalid A64 cache string");
        return {reinterpret_cast<const char*>(value.data()), value.size()};
    }
    Reference Ref() {
        Reference ref; ref.image = Integer<std::uint32_t>(); ref.offset = Integer<std::uint64_t>();
        ref.flags = Integer<std::uint64_t>(); return ref;
    }
    template<class T, class F> std::vector<T> Vector(F read) {
        const auto count = Integer<std::uint64_t>();
        if (count > maximum_items || count > bytes.size()) throw std::runtime_error("Invalid A64 cache item count");
        std::vector<T> result; result.reserve(count);
        for (std::uint64_t i = 0; i < count; ++i) result.push_back(read());
        return result;
    }
};
}

std::uint64_t Hash(std::span<const std::uint8_t> bytes, std::uint64_t seed) {
    for (auto byte : bytes) { seed ^= byte; seed *= 1099511628211ULL; }
    return seed;
}

HostImage HostImageAt(std::uint64_t address) {
    HostImage image;
#ifdef _WIN32
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(address), &module))
        throw std::runtime_error("A64 helper is outside a loaded host image");
    std::wstring name(32768, L'\0');
    const auto length = GetModuleFileNameW(module, name.data(), static_cast<DWORD>(name.size()));
    if (!length || length == name.size()) throw std::runtime_error("Cannot identify A64 host image");
    name.resize(length);
    const std::filesystem::path path(name);
    image.name = path.filename().string();
    image.identity = FileIdentity(path);
    image.base = reinterpret_cast<std::uint64_t>(module);
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(module);
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(image.base + dos->e_lfanew);
    image.size = nt->OptionalHeader.SizeOfImage;
#else
    Dl_info info{};
    if (!dladdr(reinterpret_cast<const void*>(address), &info) || !info.dli_fbase || !info.dli_fname)
        throw std::runtime_error("A64 helper is outside a loaded host image");
    const std::filesystem::path path(info.dli_fname);
    image.name = path.filename().string();
    image.identity = FileIdentity(path);
    image.base = reinterpret_cast<std::uint64_t>(info.dli_fbase);
    struct Query { std::uint64_t base, size; } query{image.base, 0};
    dl_iterate_phdr(+[](dl_phdr_info* entry, std::size_t, void* opaque) {
        auto& q = *static_cast<Query*>(opaque);
        if (entry->dlpi_addr != q.base) return 0;
        for (std::size_t i = 0; i < entry->dlpi_phnum; ++i)
            if (entry->dlpi_phdr[i].p_type == PT_LOAD)
                q.size = std::max(q.size, entry->dlpi_phdr[i].p_vaddr + entry->dlpi_phdr[i].p_memsz);
        return 1;
    }, &query);
    image.size = query.size;
#endif
    if (address < image.base || address - image.base >= image.size)
        throw std::runtime_error("A64 helper offset is outside its host image");
    return image;
}

HostImage ResolveHostImage(const HostImage& saved) {
#ifdef _WIN32
    const auto module = GetModuleHandleW(std::filesystem::path(saved.name).c_str());
    if (!module) throw std::runtime_error("A64 cache host image is not loaded");
    auto image = HostImageAt(reinterpret_cast<std::uint64_t>(module));
#else
    struct Query { const std::string* name; std::uint64_t base = 0; } query{&saved.name};
    dl_iterate_phdr(+[](dl_phdr_info* entry, std::size_t, void* opaque) {
        auto& q = *static_cast<Query*>(opaque);
        std::filesystem::path path = entry->dlpi_name;
        if (path.empty()) path = "/proc/self/exe";
        if (std::filesystem::canonical(path).filename() != *q.name) return 0;
        q.base = entry->dlpi_addr; return 1;
    }, &query);
    if (!query.base) throw std::runtime_error("A64 cache host image is not loaded");
    auto image = HostImageAt(query.base);
#endif
    if (image.identity != saved.identity || image.size != saved.size)
        throw std::runtime_error("A64 cache host image has a different build");
    return image;
}

std::string RuntimeIdentity() {
    static const std::string identity = [] {
        const auto image = HostImageAt(reinterpret_cast<std::uint64_t>(&RuntimeIdentity));
        return image.name + ":" + image.identity;
    }();
    return identity;
}

std::string Key(const Region& region) {
    Writer writer;
    writer.String(region.runtime_identity); writer.Integer(region.contract); writer.Integer(region.features);
    if (region.blocks.empty()) throw std::runtime_error("Empty A64 cache region");
    const auto& first = region.blocks.front().location;
    writer.String(region.images.at(first.image)); writer.Integer(first.offset); writer.Integer(first.flags);
    std::ostringstream name; name << std::hex << std::setfill('0') << std::setw(16) << Hash(writer.bytes);
    return name.str();
}

void Write(const std::filesystem::path& path, const Region& region) {
    Writer w;
    w.String(region.runtime_identity); w.Integer(region.contract); w.Integer(region.features);
    w.Integer(region.capacity); w.Integer(region.prelude); w.Integer(region.data_offset);
    w.Vector(region.images, [&](const auto& value) { w.String(value); });
    w.Vector(region.hosts, [&](const auto& value) { w.String(value.name); w.String(value.identity); w.Integer(value.size); });
    w.Bytes(region.data); w.Bytes(region.body);
    w.Vector(region.relocations, [&](const auto& value) { w.Integer(value.offset); w.Integer(value.kind); w.Ref(value.target); });
    w.Vector(region.blocks, [&](const auto& value) {
        w.Ref(value.location); w.Integer(value.entry); w.Integer(value.size);
        w.Vector(value.words, [&](const auto& word) { w.Ref(word.address); w.Integer(word.instruction); });
    });
    w.Vector(region.links, [&](const auto& value) { w.Ref(value.location); w.Integer(value.cell); w.Integer(value.fallback); });
    w.Vector(region.faults, [&](const auto& value) { w.Integer(value.rip); w.Integer(value.resume); w.Integer(value.callback); });
    const auto checksum = Hash(w.bytes);
    Writer header; header.bytes.insert(header.bytes.end(), magic.begin(), magic.end());
    header.Integer<std::uint64_t>(w.bytes.size()); header.Integer(checksum);
    static std::atomic<std::uint64_t> sequence{0};
    std::ostringstream suffix;
#ifdef _WIN32
    suffix << ".tmp-" << GetCurrentProcessId();
#else
    suffix << ".tmp-" << getpid();
#endif
    suffix << '-' << sequence.fetch_add(1, std::memory_order_relaxed);
    auto temporary = path; temporary += suffix.str();
    std::filesystem::create_directories(path.parent_path());
    try {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(header.bytes.data()), header.bytes.size());
        output.write(reinterpret_cast<const char*>(w.bytes.data()), w.bytes.size());
        output.close();
        if (!output) throw std::runtime_error("Cannot write A64 machine-code cache");
#ifdef _WIN32
        if (!MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING))
            throw std::runtime_error("Cannot publish A64 machine-code cache");
#else
        std::filesystem::rename(temporary, path);
#endif
    } catch (...) {
        std::error_code error; std::filesystem::remove(temporary, error); throw;
    }
}

Region Read(const std::filesystem::path& path, std::uint64_t maximum_bytes) {
    const auto size = std::filesystem::file_size(path);
    if (size < 24 || size > maximum_bytes) throw std::runtime_error("Invalid A64 machine-code cache size");
    std::vector<std::uint8_t> bytes(size);
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!input || !std::equal(magic.begin(), magic.end(), bytes.begin()))
        throw std::runtime_error("Invalid A64 machine-code cache header");
    Reader header{std::span(bytes).subspan(8)};
    if (header.Integer<std::uint64_t>() != size - 24) throw std::runtime_error("Invalid A64 cache payload size");
    const auto checksum = header.Integer<std::uint64_t>();
    if (Hash(header.bytes) != checksum) throw std::runtime_error("Damaged A64 machine-code cache");
    Reader r{header.bytes}; Region region;
    region.runtime_identity = r.String(); region.contract = r.Integer<std::uint64_t>(); region.features = r.Integer<std::uint64_t>();
    region.capacity = r.Integer<std::uint64_t>(); region.prelude = r.Integer<std::uint64_t>(); region.data_offset = r.Integer<std::uint64_t>();
    region.images = r.Vector<std::string>([&] { return r.String(); });
    region.hosts = r.Vector<HostImage>([&] { HostImage value; value.name = r.String(); value.identity = r.String(); value.size = r.Integer<std::uint64_t>(); return value; });
    region.data = r.Bytes(); region.body = r.Bytes();
    region.relocations = r.Vector<Relocation>([&] { Relocation value; value.offset = r.Integer<std::uint64_t>(); value.kind = r.Integer<std::uint8_t>(); value.target = r.Ref(); return value; });
    region.blocks = r.Vector<Block>([&] {
        Block value; value.location = r.Ref(); value.entry = r.Integer<std::uint64_t>(); value.size = r.Integer<std::uint64_t>();
        value.words = r.Vector<Word>([&] { Word word; word.address = r.Ref(); word.instruction = r.Integer<std::uint32_t>(); return word; });
        return value;
    });
    region.links = r.Vector<Link>([&] { Link value; value.location = r.Ref(); value.cell = r.Integer<std::uint64_t>(); value.fallback = r.Integer<std::uint64_t>(); return value; });
    region.faults = r.Vector<Fault>([&] { Fault value; value.rip = r.Integer<std::uint64_t>(); value.resume = r.Integer<std::uint64_t>(); value.callback = r.Integer<std::uint64_t>(); return value; });
    if (!r.bytes.empty()) throw std::runtime_error("Trailing A64 machine-code cache data");
    return region;
}

void Trim(const std::filesystem::path& directory, std::uint64_t maximum_bytes) {
    struct Entry { std::filesystem::path path; std::uint64_t size; std::filesystem::file_time_type time; };
    std::vector<Entry> files; std::uint64_t total = 0;
    for (const auto& file : std::filesystem::directory_iterator(directory)) {
        if (!file.is_regular_file() || file.path().extension() != ".a64c") continue;
        const auto size = file.file_size(); total += size;
        files.push_back({file.path(), size, file.last_write_time()});
    }
    std::sort(files.begin(), files.end(), [](const auto& a, const auto& b) { return a.time < b.time; });
    for (const auto& file : files) {
        if (total <= maximum_bytes) break;
        std::error_code error;
        if (std::filesystem::remove(file.path, error)) total -= file.size;
    }
}
} // namespace Dynarmic::A64::Persistent
