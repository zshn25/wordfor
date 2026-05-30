// WordFor C-ABI bridge header.
// Auto-generated from Rust FFI exports. Include in Swift bridging header or JNI wrapper.

#ifndef WORDFOR_H
#define WORDFOR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the engine with a data directory and mode.
// mode: "full", "binary", or "lite"
// Returns 0 on success, -1 on error.
int32_t wordfor_init(const char *data_dir, const char *mode);

// Search using a pre-computed query vector (float32 array).
// Returns a JSON string that must be freed with wordfor_free_string().
// Returns NULL on error.
char *wordfor_search_vector(const float *qvec, size_t qvec_len);

// Get the number of loaded dictionary entries.
uint32_t wordfor_entry_count(void);

// Free a string returned by wordfor_search_vector.
void wordfor_free_string(char *s);

#ifdef __cplusplus
}
#endif

#endif // WORDFOR_H
