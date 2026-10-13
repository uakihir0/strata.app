// src/metal/engine/mtp_draft_tail_dispatch.cpp - ordered runtime MTP stages 21–22.
#include "strata/metal/mtp_dispatch.hpp"

#include "strata/metal/encoder.hpp"

namespace strata::metal {
namespace {

bool same_view(BufferRef left, BufferRef right) noexcept {
    return left && right && left.handle == right.handle &&
           left.contents == right.contents &&
           left.gpu_address == right.gpu_address &&
           left.offset == right.offset && left.length == right.length;
}

bool same_view(BufferRef writable, ReadOnlyBufferRef readable) noexcept {
    return writable && readable &&
           writable.contents == readable.contents() &&
           writable.gpu_address == readable.gpu_address() &&
           writable.offset == readable.offset() &&
           writable.length == readable.length();
}

bool is_empty_map_selection(
    const MtpDraftVocabularyMap& map) noexcept {
    return !map.rows() && map.row_count() == 0u &&
           map.full_head_size() == 0u;
}

bool has_valid_map_selection(
    const MtpDraftVocabularyMap& map) noexcept {
    return static_cast<bool>(map) || is_empty_map_selection(map);
}

bool same_map(const MtpDraftVocabularyMap& left,
              const MtpDraftVocabularyMap& right) noexcept {
    if (static_cast<bool>(left) != static_cast<bool>(right)) return false;
    if (!left) {
        return is_empty_map_selection(left) &&
               is_empty_map_selection(right);
    }
    const ReadOnlyBufferRef left_rows = left.rows();
    const ReadOnlyBufferRef right_rows = right.rows();
    return left.row_count() == right.row_count() &&
           left.full_head_size() == right.full_head_size() &&
           left_rows && right_rows &&
           left_rows.contents() == right_rows.contents() &&
           left_rows.gpu_address() == right_rows.gpu_address() &&
           left_rows.offset() == right_rows.offset() &&
           left_rows.length() == right_rows.length();
}

bool validate_mtp_draft_tail_bindings(
    const MtpDraftHeadBuffers& head,
    const MtpDraftSamplingBuffers& sampling, std::string& error) {
    if (!same_view(sampling.params, head.params) ||
        !same_view(head.logits, sampling.logits)) {
        error =
            "MTP draft tail requires identical parameter and logits views";
        return false;
    }
    if (!has_valid_map_selection(head.draft_vocabulary) ||
        !has_valid_map_selection(sampling.draft_vocabulary)) {
        error =
            "MTP draft tail requires live validated vocabulary-map tokens";
        return false;
    }
    if (!same_map(head.draft_vocabulary,
                  sampling.draft_vocabulary)) {
        error =
            "MTP draft tail requires the same validated vocabulary-map token";
        return false;
    }
    return true;
}

}  // namespace

bool encode_mtp_draft_tail(Context& context, Encoder& encoder,
                           MatVecWeightFormat weight_format,
                           std::uint32_t head_vocab_size,
                           const MtpDraftHeadBuffers& head,
                           const MtpDraftSamplingBuffers& sampling,
                           std::string& error) {
    error.clear();
    if (!validate_mtp_draft_tail_bindings(head, sampling, error)) {
        return false;
    }
    if (!encode_mtp_draft_head(context, encoder, weight_format,
                               head_vocab_size, head, error) ||
        !encode_mtp_draft_sampling(context, encoder, head_vocab_size,
                                   sampling, error)) {
        return false;
    }
    error.clear();
    return true;
}

bool encode_mtp_draft_tail(Context& context, CommandBuffer& command_buffer,
                           MatVecWeightFormat weight_format,
                           std::uint32_t head_vocab_size,
                           const MtpDraftHeadBuffers& head,
                           const MtpDraftSamplingBuffers& sampling,
                           std::string& error) {
    error.clear();
    return Encoder::encode(
        context, command_buffer, "MTP draft tail stages 21–22",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_mtp_draft_tail(context, encoder, weight_format,
                                         head_vocab_size, head, sampling,
                                         encode_error);
        },
        error);
}

}  // namespace strata::metal
