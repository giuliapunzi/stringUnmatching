// RG-July2021: open a file using mmap

// TODO: enable to open multiple files

#include <iostream>

// for mmap:
#include <sys/mman.h>
#include <sys/stat.h>

/* MMAP */

FILE *fd;

void handle_error(const char* msg) {
    perror(msg); 
    exit(255);
}

const char* map_file(const char* fname, size_t& length)
{
    fd = fopen(fname, "r");
    if (fd == NULL)
        handle_error("open");

    // obtain file size
    struct stat sb;
    if (fstat(fileno(fd), &sb) == -1)
        handle_error("fstat");

    length = sb.st_size;

    const char* addr = static_cast<const char*>(mmap(NULL, length, PROT_READ, MAP_SHARED, fileno(fd), 0u));
    if (addr == MAP_FAILED)
        handle_error("mmap");

    // TODO close fd at some point in time, call munmap(...)
    return addr;
}

void unmap_file(const char* addr, size_t length)
{
    munmap(const_cast<char *>(addr), length);
    fclose(fd);
}

