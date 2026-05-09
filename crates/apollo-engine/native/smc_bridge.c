/*
 * smc_bridge.c — C shim for SMC (System Management Controller) direct reads.
 *
 * SMC provides sub-100µs access to:
 *   - PSTR: System total power (watts, real-time)
 *   - MSLD: Lid state (open/closed)
 *   - CLSP/CLWK: Last sleep/wake timestamps (microseconds)
 *   - B0TE/B0TF: Battery time to empty/full (minutes)
 *   - PDTR: Charger power delivery (watts)
 *
 * Architecture: 2-step read via IOConnectCallStructMethod.
 *   Step 1: kSMCGetKeyInfo (data8=9) → type/size metadata
 *   Step 2: kSMCReadKey    (data8=5) → raw bytes
 *
 * Follows the same pattern as ioreport_bridge.c: plain C functions
 * callable from Rust via FFI.
 *
 * Requires: IOKit.framework (already linked by IOReport).
 */

#include <IOKit/IOKitLib.h>
#include <stdint.h>
#include <string.h>

/* ── SMC constants ─────────────────────────────────────────────────────────── */

#define SMC_KERNEL_INDEX    2
#define SMC_CMD_READ_KEY    5
#define SMC_CMD_GET_KEYINFO 9

/* ── SMC data structures ───────────────────────────────────────────────────── */

typedef struct {
    char    major;
    char    minor;
    char    build;
    char    reserved;
    uint16_t release;
} SMCKeyData_vers_t;

typedef struct {
    uint32_t dataSize;
    uint32_t dataType;
    uint8_t  dataAttributes;
} SMCKeyData_keyInfo_t;

typedef struct {
    uint32_t            key;
    SMCKeyData_vers_t   vers;
    uint16_t            pLimitData;
    uint16_t            keyInfo_pLimitData;
    SMCKeyData_keyInfo_t keyInfo;
    uint8_t             result;
    uint8_t             status;
    uint8_t             data8;
    uint32_t            data32;
    uint8_t             bytes[32];
} SMCKeyData_t;

/* ── Bridge functions ──────────────────────────────────────────────────────── */

/*
 * apollo_smc_open — open a connection to the SMC service.
 * Returns the IOKit connection handle, or 0 on failure.
 */
io_connect_t apollo_smc_open(void) {
    io_service_t service = IOServiceGetMatchingService(
        kIOMainPortDefault,
        IOServiceMatching("AppleSMC")
    );
    if (service == 0) return 0;

    io_connect_t conn = 0;
    kern_return_t kr = IOServiceOpen(service, mach_task_self(), 0, &conn);
    IOObjectRelease(service);

    return (kr == KERN_SUCCESS) ? conn : 0;
}

/*
 * apollo_smc_close — close the SMC connection.
 */
void apollo_smc_close(io_connect_t conn) {
    if (conn) IOServiceClose(conn);
}

/*
 * apollo_smc_read_key — read a 4-char SMC key.
 *
 * conn:      connection from apollo_smc_open
 * key:       4-byte key (e.g., 'PSTR' as uint32)
 * out_bytes: buffer to receive raw value bytes (must be ≥32 bytes)
 * out_size:  receives actual data size
 * out_type:  receives data type code (e.g., 'flt ' for float)
 *
 * Returns 0 on success, non-zero on failure.
 */
int apollo_smc_read_key(
    io_connect_t conn,
    uint32_t key,
    uint8_t *out_bytes,
    uint32_t *out_size,
    uint32_t *out_type)
{
    if (!conn) return -1;

    SMCKeyData_t input;
    SMCKeyData_t output;
    size_t input_size = sizeof(SMCKeyData_t);
    size_t output_size = sizeof(SMCKeyData_t);

    /* Step 1: Get key info (type + size) */
    memset(&input, 0, sizeof(input));
    memset(&output, 0, sizeof(output));
    input.key = key;
    input.data8 = SMC_CMD_GET_KEYINFO;

    kern_return_t kr = IOConnectCallStructMethod(
        conn, SMC_KERNEL_INDEX,
        &input, input_size,
        &output, &output_size
    );
    if (kr != KERN_SUCCESS) return (int)kr;

    /* Step 2: Read the value */
    uint32_t data_size = output.keyInfo.dataSize;
    uint32_t data_type = output.keyInfo.dataType;

    memset(&input, 0, sizeof(input));
    input.key = key;
    input.keyInfo.dataSize = data_size;
    input.data8 = SMC_CMD_READ_KEY;

    output_size = sizeof(SMCKeyData_t);
    memset(&output, 0, sizeof(output));

    kr = IOConnectCallStructMethod(
        conn, SMC_KERNEL_INDEX,
        &input, input_size,
        &output, &output_size
    );
    if (kr != KERN_SUCCESS) return (int)kr;

    /* Copy results out */
    if (out_bytes) {
        uint32_t copy_size = data_size < 32 ? data_size : 32;
        memcpy(out_bytes, output.bytes, copy_size);
    }
    if (out_size) *out_size = data_size;
    if (out_type) *out_type = data_type;

    return 0;
}
