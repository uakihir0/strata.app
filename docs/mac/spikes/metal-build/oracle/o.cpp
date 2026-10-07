#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
int main() {
  const auto* t = ggml_get_type_traits(GGML_TYPE_IQ2_XS);
  const auto* c = ggml_get_type_traits_cpu(GGML_TYPE_IQ2_XS);
  std::printf("iq2_xs blck=%lld size=%zu vec_dot=%d\n", (long long)t->blck_size, t->type_size, c->vec_dot != nullptr);
  return 0;
}
