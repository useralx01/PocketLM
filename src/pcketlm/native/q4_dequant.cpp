#include <cstdint>
#include <cstring>

static float fp16_to_float(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x03FFu;
    uint32_t out = 0;

    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            uint32_t m = mant;
            int e = -14;
            while ((m & 0x0400u) == 0) {
                m <<= 1;
                --e;
            }
            m &= 0x03FFu;
            const uint32_t out_exp = static_cast<uint32_t>(e + 127);
            out = sign | (out_exp << 23) | (m << 13);
        }
    } else if (exp == 0x1Fu) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        const uint32_t out_exp = exp + (127 - 15);
        out = sign | (out_exp << 23) | (mant << 13);
    }

    float value = 0.0f;
    std::memcpy(&value, &out, sizeof(value));
    return value;
}

static uint16_t float_to_fp16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t mant = bits & 0x007FFFFFu;
    int exp = static_cast<int>((bits >> 23) & 0xFFu);

    if (exp == 255) {
        if (mant == 0) {
            return static_cast<uint16_t>(sign | 0x7C00u);
        }
        return static_cast<uint16_t>(sign | 0x7C00u | (mant >> 13) | 1u);
    }

    exp = exp - 127 + 15;
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant |= 0x00800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t half_mant = mant >> shift;
        const uint32_t round_bit = (mant >> (shift - 1)) & 1u;
        const uint32_t sticky = mant & ((1u << (shift - 1)) - 1u);
        if (round_bit && (sticky || (half_mant & 1u))) {
            ++half_mant;
        }
        return static_cast<uint16_t>(sign | half_mant);
    }

    uint32_t half_exp = static_cast<uint32_t>(exp) << 10;
    uint32_t half_mant = mant >> 13;
    const uint32_t round_bits = mant & 0x1FFFu;
    if (round_bits > 0x1000u || (round_bits == 0x1000u && (half_mant & 1u))) {
        ++half_mant;
        if (half_mant == 0x0400u) {
            half_mant = 0;
            half_exp += 0x0400u;
            if (half_exp >= 0x7C00u) {
                return static_cast<uint16_t>(sign | 0x7C00u);
            }
        }
    }
    return static_cast<uint16_t>(sign | half_exp | half_mant);
}

extern "C" __declspec(dllexport) void q4_dequant_to_fp16(
    const uint8_t* packed,
    const uint16_t* scales,
    uint16_t* out_fp16,
    int64_t num_channels,
    int64_t channel_size
) {
    for (int64_t c = 0; c < num_channels; ++c) {
        const float scale = fp16_to_float(scales[c]);
        const int64_t channel_offset = c * channel_size;
        for (int64_t i = 0; i < channel_size; ++i) {
            const int64_t value_index = channel_offset + i;
            const uint8_t byte = packed[value_index / 2];
            uint8_t nibble = (value_index % 2 == 0) ? (byte & 0x0Fu) : ((byte >> 4) & 0x0Fu);
            int8_t q = static_cast<int8_t>(nibble);
            if (q >= 8) {
                q = static_cast<int8_t>(q - 16);
            }
            out_fp16[value_index] = float_to_fp16(static_cast<float>(q) * scale);
        }
    }
}
