#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

#include <tokenizers_cpp.h>
#include <tokenizers_c.h>

namespace py = pybind11;
#define CPP_PRINT(msg) py::print("[C++] " + std::string(msg))

// C API wrapper class
class TokenizerC {
public:
    TokenizerHandle handle;

    TokenizerC(TokenizerHandle h) : handle(h) {}

    ~TokenizerC() {
        if (handle) {
            tokenizers_free(handle);
        }
    }

    std::vector<int32_t> Encode(const std::string& text, bool add_special_token = true) {
        TokenizerEncodeResult result;
        tokenizers_encode(handle, text.c_str(), text.length(), add_special_token ? 1 : 0, &result);

        std::vector<int32_t> token_ids(result.token_ids, result.token_ids + result.len);
        tokenizers_free_encode_results(&result, 1);

        return token_ids;
    }

    std::vector<std::vector<int32_t>> EncodeBatch(const std::vector<std::string>& texts, bool add_special_token = true) {
        std::vector<const char*> data_ptrs;
        std::vector<size_t> data_lens;

        for (const auto& text : texts) {
            data_ptrs.push_back(text.c_str());
            data_lens.push_back(text.length());
        }

        std::vector<TokenizerEncodeResult> results(texts.size());
        tokenizers_encode_batch(handle, data_ptrs.data(), data_lens.data(), texts.size(),
                              add_special_token ? 1 : 0, results.data());

        std::vector<std::vector<int32_t>> batch_ids;
        for (size_t i = 0; i < texts.size(); ++i) {
            batch_ids.emplace_back(results[i].token_ids, results[i].token_ids + results[i].len);
        }

        tokenizers_free_encode_results(results.data(), texts.size());

        return batch_ids;
    }

    std::string Decode(const std::vector<int32_t>& ids, bool skip_special_token = true) {
        tokenizers_decode(handle, reinterpret_cast<const uint32_t*>(ids.data()), ids.size(),
                         skip_special_token ? 1 : 0);

        const char* data;
        size_t len;
        tokenizers_get_decode_str(handle, &data, &len);

        return std::string(data, len);
    }

    size_t GetVocabSize() {
        size_t size;
        tokenizers_get_vocab_size(handle, &size);
        return size;
    }

    std::string IdToToken(uint32_t token_id) {
        const char* data;
        size_t len;
        tokenizers_id_to_token(handle, token_id, &data, &len);
        return std::string(data, len);
    }

    int32_t TokenToId(const std::string& token) {
        int32_t id;
        tokenizers_token_to_id(handle, token.c_str(), token.length(), &id);
        return id;
    }

    static TokenizerC* FromBlobJSON(const std::string& json_blob) {
        TokenizerHandle handle = tokenizers_new_from_str(json_blob.c_str(), json_blob.length());
        return handle ? new TokenizerC(handle) : nullptr;
    }

    static TokenizerC* FromBlobByteLevelBPE(const std::string& vocab_blob, const std::string& merges_blob,
                                           const std::string& added_tokens = "") {
        TokenizerHandle handle = byte_level_bpe_tokenizers_new_from_str(
            vocab_blob.c_str(), vocab_blob.length(),
            merges_blob.c_str(), merges_blob.length(),
            added_tokens.c_str(), added_tokens.length());
        return handle ? new TokenizerC(handle) : nullptr;
    }
};

PYBIND11_MODULE(tokenizers_cpp, m) {
    m.doc() = "C++ tokenizers bindings";

    py::class_<tokenizers::Tokenizer>(m, "TokenizerCPP")
        .def("Encode", &tokenizers::Tokenizer::Encode)
        .def("EncodeBatch", &tokenizers::Tokenizer::EncodeBatch)
        .def("Decode", &tokenizers::Tokenizer::Decode)
        .def("GetVocabSize", &tokenizers::Tokenizer::GetVocabSize)
        .def("IdToToken", &tokenizers::Tokenizer::IdToToken)
        .def("TokenToId", &tokenizers::Tokenizer::TokenToId)
        .def_static("FromBlobJSON", &tokenizers::Tokenizer::FromBlobJSON);

    py::class_<TokenizerC>(m, "TokenizerC")
        .def("Encode", &TokenizerC::Encode, py::arg("text"), py::arg("add_special_token") = true)
        .def("EncodeBatch", &TokenizerC::EncodeBatch, py::arg("texts"), py::arg("add_special_token") = true)
        .def("Decode", &TokenizerC::Decode, py::arg("ids"), py::arg("skip_special_token") = true)
        .def("GetVocabSize", &TokenizerC::GetVocabSize)
        .def("IdToToken", &TokenizerC::IdToToken)
        .def("TokenToId", &TokenizerC::TokenToId)
        .def_static("FromBlobJSON", &TokenizerC::FromBlobJSON)
        .def_static("FromBlobByteLevelBPE", &TokenizerC::FromBlobByteLevelBPE,
                   py::arg("vocab_blob"), py::arg("merges_blob"), py::arg("added_tokens") = "");
}

