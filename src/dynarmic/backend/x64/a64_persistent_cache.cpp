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

struct File {
    explicit File(const std::filesystem::path& path) {
#ifdef _WIN32
        handle = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                             nullptr, OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS, nullptr);
        LARGE_INTEGER length{};
        if (handle == INVALID_HANDLE_VALUE || !GetFileSizeEx(handle, &length) || length.QuadPart < 0) {
            if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle);
            throw std::runtime_error("Cannot open A64 machine-code cache");
        }
        size = static_cast<std::uint64_t>(length.QuadPart);
#else
        input.open(path, std::ios::binary | std::ios::ate);
        const auto length = input.tellg();
        if (!input || length < 0) throw std::runtime_error("Cannot open A64 machine-code cache");
        size = static_cast<std::uint64_t>(length);
#endif
    }
    ~File() {
#ifdef _WIN32
        CloseHandle(handle);
#endif
    }
    std::vector<std::uint8_t> Read(std::uint64_t offset, std::uint64_t length) {
        if (offset > size || length > size - offset || length > std::numeric_limits<std::size_t>::max() ||
            length > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()))
            throw std::runtime_error("Invalid A64 cache file range");
        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length));
        if (bytes.empty()) return bytes;
        std::lock_guard lock(mutex);
#ifdef _WIN32
        LARGE_INTEGER position{}; position.QuadPart = static_cast<LONGLONG>(offset);
        if (!SetFilePointerEx(handle, position, nullptr, FILE_BEGIN))
            throw std::runtime_error("Cannot seek A64 machine-code cache");
        std::size_t done = 0;
        while (done < bytes.size()) {
            const auto chunk = static_cast<DWORD>(std::min<std::size_t>(bytes.size() - done, MAXDWORD));
            DWORD count = 0;
            if (!ReadFile(handle, bytes.data() + done, chunk, &count, nullptr) || count != chunk)
                throw std::runtime_error("Cannot read A64 machine-code cache");
            done += count;
        }
#else
        input.clear();
        input.seekg(static_cast<std::streamoff>(offset));
        input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(length));
        if (!input) throw std::runtime_error("Cannot read A64 machine-code cache");
#endif
        return bytes;
    }
    std::uint64_t size = 0;
    std::mutex mutex;
#ifdef _WIN32
    HANDLE handle = INVALID_HANDLE_VALUE;
#else
    std::ifstream input;
#endif
};

namespace {
constexpr std::array<std::uint8_t, 8> magic{'A', '2', 'E', 'A', '6', '4', 'C', 3};
constexpr std::size_t maximum_items = 4 * 1024 * 1024;
constexpr std::uint64_t header_size = 32;
constexpr std::uint64_t maximum_index_bytes = 16 * 1024 * 1024;
constexpr std::uint64_t maximum_data_bytes = 256 * 1024;

#ifdef _WIN32
void Publish(const std::filesystem::path& source, const std::filesystem::path& target) {
    const auto name = std::filesystem::absolute(target).native();
    const auto bytes = offsetof(FILE_RENAME_INFO, FileName) + (name.size() + 1) * sizeof(wchar_t);
    std::vector<std::uint8_t> storage(bytes);
    auto* rename = reinterpret_cast<FILE_RENAME_INFO*>(storage.data());
    rename->Flags = FILE_RENAME_FLAG_REPLACE_IF_EXISTS | FILE_RENAME_FLAG_POSIX_SEMANTICS;
    rename->FileNameLength = static_cast<DWORD>(name.size() * sizeof(wchar_t));
    std::memcpy(rename->FileName, name.c_str(), (name.size() + 1) * sizeof(wchar_t));
    const auto handle = CreateFileW(source.c_str(), DELETE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                    nullptr, OPEN_EXISTING, 0, nullptr);
    if (handle == INVALID_HANDLE_VALUE) throw std::runtime_error("Cannot open A64 cache for publication");
    const auto success = SetFileInformationByHandle(handle, FileRenameInfoEx, rename, static_cast<DWORD>(bytes));
    const auto error = GetLastError();
    CloseHandle(handle);
    if (!success) throw std::runtime_error("Cannot publish A64 machine-code cache: " + std::to_string(error));
}

bool Remove(const std::filesystem::path& path) {
    const auto handle = CreateFileW(path.c_str(), DELETE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                    nullptr, OPEN_EXISTING, 0, nullptr);
    if (handle == INVALID_HANDLE_VALUE) return false;
    FILE_DISPOSITION_INFO_EX disposition{FILE_DISPOSITION_FLAG_DELETE | FILE_DISPOSITION_FLAG_POSIX_SEMANTICS};
    const auto success = SetFileInformationByHandle(handle, FileDispositionInfoEx, &disposition, sizeof(disposition));
    CloseHandle(handle);
    return success != 0;
}
#endif

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
        const auto size = Integer<std::uint64_t>();
        if (size > 65536 || size > bytes.size()) throw std::runtime_error("Invalid A64 cache string");
        std::string value(reinterpret_cast<const char*>(bytes.data()), static_cast<std::size_t>(size));
        bytes = bytes.subspan(size);
        return value;
    }
    Reference Ref() {
        Reference ref; ref.image = Integer<std::uint32_t>(); ref.offset = Integer<std::uint64_t>();
        ref.flags = Integer<std::uint64_t>(); return ref;
    }
    template<class T, class F> std::vector<T> Vector(F read, std::size_t minimum_bytes = 1) {
        const auto count = Integer<std::uint64_t>();
        if (count > maximum_items || count > bytes.size() / minimum_bytes) throw std::runtime_error("Invalid A64 cache item count");
        std::vector<T> result; result.reserve(count);
        for (std::uint64_t i = 0; i < count; ++i) result.push_back(read());
        return result;
    }
};

Writer EncodeIndex(const Index& index) {
    Writer w;
    w.String(index.runtime_identity); w.Integer(index.contract); w.Integer(index.features);
    w.Integer(index.capacity); w.Integer(index.prelude); w.Integer(index.data_offset); w.Integer(index.body_end);
    w.Vector(index.images, [&](const auto& value) { w.String(value); });
    w.Vector(index.hosts, [&](const auto& value) { w.String(value.name); w.String(value.identity); w.Integer(value.size); });
    w.Integer(index.data_file_offset); w.Integer(index.data_size); w.Integer(index.data_checksum);
    w.Vector(index.blocks, [&](const auto& value) {
        w.Ref(value.location); w.Integer(value.entry); w.Integer(value.size);
        w.Integer(value.record_offset); w.Integer(value.record_size); w.Integer(value.record_checksum);
    });
    return w;
}

Index ReadFront(std::shared_ptr<File> file, std::uint64_t maximum_bytes) {
    if (file->size < header_size || file->size > maximum_bytes)
        throw std::runtime_error("Invalid A64 machine-code cache size");
    const auto header_bytes = file->Read(0, header_size);
    if (!std::equal(magic.begin(), magic.end(), header_bytes.begin()))
        throw std::runtime_error("Invalid A64 machine-code cache header");
    Reader header{std::span(header_bytes).subspan(magic.size())};
    const auto index_size = header.Integer<std::uint64_t>();
    const auto index_checksum = header.Integer<std::uint64_t>();
    const auto file_size = header.Integer<std::uint64_t>();
    if (file_size != file->size || index_size > maximum_index_bytes || index_size > file_size - header_size)
        throw std::runtime_error("Invalid A64 cache section sizes");
    const auto bytes = file->Read(header_size, index_size);
    if (Hash(bytes) != index_checksum)
        throw std::runtime_error("Damaged A64 cache index");
    Reader reader{bytes};
    Index index;
    index.file = std::move(file); index.file_size = file_size;
    index.read_bytes = header_size + index_size;
    index.runtime_identity = reader.String();
    index.contract = reader.Integer<std::uint64_t>();
    index.features = reader.Integer<std::uint64_t>();
    index.capacity = reader.Integer<std::uint64_t>();
    index.prelude = reader.Integer<std::uint64_t>();
    index.data_offset = reader.Integer<std::uint64_t>();
    index.body_end = reader.Integer<std::uint64_t>();
    index.images = reader.Vector<std::string>([&] { return reader.String(); }, sizeof(std::uint64_t));
    index.hosts = reader.Vector<HostImage>([&] {
        HostImage value; value.name = reader.String(); value.identity = reader.String(); value.size = reader.Integer<std::uint64_t>();
        return value;
    }, 3 * sizeof(std::uint64_t));
    index.data_file_offset = reader.Integer<std::uint64_t>();
    index.data_size = reader.Integer<std::uint64_t>();
    index.data_checksum = reader.Integer<std::uint64_t>();
    if (index.prelude > index.capacity || index.body_end < index.prelude || index.body_end > index.capacity ||
        index.data_offset > index.prelude || index.data_size > index.prelude - index.data_offset ||
        index.data_size > maximum_data_bytes || index.data_file_offset != index.read_bytes ||
        index.data_size > file_size - index.data_file_offset)
        throw std::runtime_error("Invalid A64 cache region layout");
    auto next_record = index.data_file_offset + index.data_size;
    auto block_end = index.prelude;
    index.blocks = reader.Vector<BlockIndex>([&] {
        BlockIndex block; block.location = reader.Ref(); block.entry = reader.Integer<std::uint64_t>();
        block.size = reader.Integer<std::uint64_t>(); block.record_offset = reader.Integer<std::uint64_t>();
        block.record_size = reader.Integer<std::uint64_t>(); block.record_checksum = reader.Integer<std::uint64_t>();
        if (block.location.image >= index.images.size() || !block.size || block.entry < block_end ||
            block.entry > index.body_end || block.size > index.body_end - block.entry ||
            block.record_offset != next_record || block.record_size > file_size - next_record ||
            block.record_size < block.size || block.record_size - block.size < 100)
            throw std::runtime_error("Invalid A64 cache block index");
        block_end = block.entry + block.size;
        next_record += block.record_size;
        return block;
    }, sizeof(std::uint32_t) + 7 * sizeof(std::uint64_t));
    if (!reader.bytes.empty()) throw std::runtime_error("Trailing A64 cache index data");
    if (index.blocks.empty() || next_record != file_size) throw std::runtime_error("Invalid A64 cache record layout");
    return index;
}
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
    if (region.blocks.empty() || region.prelude > region.capacity || region.body.size() > region.capacity - region.prelude ||
        region.data_offset > region.prelude || region.data.size() > region.prelude - region.data_offset ||
        region.data.size() > maximum_data_bytes)
        throw std::runtime_error("Invalid A64 cache region layout");
    Index index;
    index.runtime_identity = region.runtime_identity; index.contract = region.contract; index.features = region.features;
    index.capacity = region.capacity; index.prelude = region.prelude; index.data_offset = region.data_offset;
    index.body_end = region.prelude + region.body.size(); index.images = region.images; index.hosts = region.hosts;
    index.data_size = region.data.size(); index.data_checksum = Hash(region.data);
    const auto sorted = [](const auto& values, auto offset) {
        std::vector<std::size_t> order(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return offset(values[a]) < offset(values[b]); });
        return order;
    };
    const auto blocks = sorted(region.blocks, [](const auto& value) { return value.entry; });
    const auto relocations = sorted(region.relocations, [](const auto& value) { return value.offset; });
    const auto links = sorted(region.links, [](const auto& value) { return value.fallback; });
    const auto faults = sorted(region.faults, [](const auto& value) { return value.rip; });
    std::size_t relocation_cursor = 0, link_cursor = 0, fault_cursor = 0;
    std::vector<Writer> records; records.reserve(blocks.size()); index.blocks.reserve(blocks.size());
    auto block_end = region.prelude;
    for (const auto ordinal : blocks) {
        const auto& block = region.blocks[ordinal];
        if (block.location.image >= region.images.size() || !block.size || block.entry < block_end ||
            block.entry > index.body_end || block.size > index.body_end - block.entry ||
            block.words.empty() || block.words.size() > 4096)
            throw std::runtime_error("Invalid A64 cache block");
        block_end = block.entry + block.size;
        Writer w;
        w.Ref(block.location); w.Integer(block.entry); w.Integer(block.size);
        w.Vector(block.words, [&](const auto& word) { w.Ref(word.address); w.Integer(word.instruction); });
        w.Bytes(std::span(region.body).subspan(block.entry - region.prelude, block.size));
        const auto write_owned = [&](const auto& values, const auto& order, auto& cursor, auto offset, auto write) {
            const auto begin = cursor;
            while (cursor < order.size() && offset(values[order[cursor]]) < block_end) {
                if (offset(values[order[cursor]]) < block.entry)
                    throw std::runtime_error("A64 cache metadata has no owning block");
                ++cursor;
            }
            w.Integer<std::uint64_t>(cursor - begin);
            for (auto i = begin; i < cursor; ++i) write(values[order[i]]);
        };
        write_owned(region.relocations, relocations, relocation_cursor,
            [](const auto& value) { return value.offset; },
            [&](const auto& value) { w.Integer(value.offset); w.Integer(value.kind); w.Ref(value.target); });
        write_owned(region.links, links, link_cursor,
            [](const auto& value) { return value.fallback; },
            [&](const auto& value) { w.Ref(value.location); w.Integer(value.cell); w.Integer(value.fallback); });
        write_owned(region.faults, faults, fault_cursor,
            [](const auto& value) { return value.rip; },
            [&](const auto& value) { w.Integer(value.rip); w.Integer(value.resume); w.Integer(value.callback); });
        index.blocks.push_back({block.location, block.entry, block.size, 0, w.bytes.size(), Hash(w.bytes)});
        records.push_back(std::move(w));
    }
    if (relocation_cursor != relocations.size() || link_cursor != links.size() || fault_cursor != faults.size())
        throw std::runtime_error("A64 cache metadata has no owning block");
    auto encoded_index = EncodeIndex(index);
    if (encoded_index.bytes.size() > maximum_index_bytes)
        throw std::runtime_error("A64 cache index exceeds size limit");
    index.data_file_offset = header_size + encoded_index.bytes.size();
    auto record_offset = index.data_file_offset + index.data_size;
    for (auto& block : index.blocks) {
        block.record_offset = record_offset;
        if (block.record_size > std::numeric_limits<std::uint64_t>::max() - record_offset)
            throw std::runtime_error("A64 cache file exceeds size limit");
        record_offset += block.record_size;
    }
    encoded_index = EncodeIndex(index);
    Writer header; header.bytes.insert(header.bytes.end(), magic.begin(), magic.end());
    header.Integer<std::uint64_t>(encoded_index.bytes.size()); header.Integer(Hash(encoded_index.bytes));
    header.Integer(record_offset);
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
        output.write(reinterpret_cast<const char*>(encoded_index.bytes.data()), encoded_index.bytes.size());
        output.write(reinterpret_cast<const char*>(region.data.data()), region.data.size());
        for (const auto& record : records)
            output.write(reinterpret_cast<const char*>(record.bytes.data()), record.bytes.size());
        output.close();
        if (!output) throw std::runtime_error("Cannot write A64 machine-code cache");
#ifdef _WIN32
        Publish(temporary, path);
#else
        std::filesystem::rename(temporary, path);
#endif
    } catch (...) {
        std::error_code error; std::filesystem::remove(temporary, error); throw;
    }
}

Index ReadIndex(const std::filesystem::path& path, std::uint64_t maximum_bytes) {
    return ReadFront(std::make_shared<File>(path), maximum_bytes);
}

std::vector<std::uint8_t> ReadData(const Index& index) {
    if (!index.file) throw std::runtime_error("A64 cache file is not open");
    auto bytes = index.file->Read(index.data_file_offset, index.data_size);
    if (Hash(bytes) != index.data_checksum) throw std::runtime_error("Damaged A64 cache shared data");
    return bytes;
}

BlockRecord ReadBlock(const Index& index, std::size_t ordinal) {
    if (!index.file || ordinal >= index.blocks.size()) throw std::runtime_error("Invalid A64 cache block ordinal");
    const auto& indexed = index.blocks[ordinal];
    const auto bytes = index.file->Read(indexed.record_offset, indexed.record_size);
    if (Hash(bytes) != indexed.record_checksum) throw std::runtime_error("Damaged A64 cache block record");
    Reader r{bytes}; BlockRecord record; record.read_bytes = bytes.size();
    auto& block = record.block;
    block.location = r.Ref(); block.entry = r.Integer<std::uint64_t>(); block.size = r.Integer<std::uint64_t>();
    if (block.location.image != indexed.location.image || block.location.offset != indexed.location.offset ||
        block.location.flags != indexed.location.flags || block.entry != indexed.entry || block.size != indexed.size)
        throw std::runtime_error("A64 cache block differs from index");
    block.words = r.Vector<Word>([&] {
        Word word; word.address = r.Ref(); word.instruction = r.Integer<std::uint32_t>();
        if (word.address.image >= index.images.size()) throw std::runtime_error("Invalid A64 cache source image");
        return word;
    }, sizeof(std::uint32_t) * 2 + sizeof(std::uint64_t) * 2);
    if (block.words.empty() || block.words.size() > 4096) throw std::runtime_error("Invalid A64 cache source words");
    record.body = r.Bytes();
    if (record.body.size() != block.size) throw std::runtime_error("Invalid A64 cache block body");
    const auto contains = [&](std::uint64_t offset) { return offset >= block.entry && offset - block.entry < block.size; };
    record.relocations = r.Vector<Relocation>([&] {
        Relocation value; value.offset = r.Integer<std::uint64_t>(); value.kind = r.Integer<std::uint8_t>(); value.target = r.Ref();
        if (!contains(value.offset) || sizeof(std::uint64_t) > block.size - (value.offset - block.entry) || value.kind > 5 ||
            (value.kind == 0 && (value.target.image >= index.hosts.size() || value.target.offset >= index.hosts[value.target.image].size)) ||
            ((value.kind == 1 || value.kind == 2 || value.kind == 5) && value.target.image >= index.images.size()))
            throw std::runtime_error("Invalid A64 cache relocation");
        return value;
    }, sizeof(std::uint8_t) + sizeof(std::uint32_t) + sizeof(std::uint64_t) * 3);
    record.links = r.Vector<Link>([&] {
        Link value; value.location = r.Ref(); value.cell = r.Integer<std::uint64_t>(); value.fallback = r.Integer<std::uint64_t>();
        if (value.location.image >= index.images.size() || !contains(value.fallback) || value.cell < index.data_offset ||
            value.cell - index.data_offset > index.data_size || sizeof(std::uint64_t) > index.data_size - (value.cell - index.data_offset) ||
            value.cell % alignof(std::uint64_t))
            throw std::runtime_error("Invalid A64 cache link");
        return value;
    }, sizeof(std::uint32_t) + sizeof(std::uint64_t) * 4);
    record.faults = r.Vector<Fault>([&] {
        Fault value; value.rip = r.Integer<std::uint64_t>(); value.resume = r.Integer<std::uint64_t>(); value.callback = r.Integer<std::uint64_t>();
        const bool prelude_callback = value.callback >= index.data_offset + index.data_size && value.callback < index.prelude;
        if (!contains(value.rip) || (value.resume && !contains(value.resume)) || (!prelude_callback && !contains(value.callback)))
            throw std::runtime_error("Invalid A64 cache fastmem fault");
        return value;
    }, sizeof(std::uint64_t) * 3);
    if (!r.bytes.empty()) throw std::runtime_error("Trailing A64 cache block data");
    return record;
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
#ifdef _WIN32
        if (Remove(file.path)) total -= file.size;
#else
        std::error_code error;
        if (std::filesystem::remove(file.path, error)) total -= file.size;
#endif
    }
}
} // namespace Dynarmic::A64::Persistent
